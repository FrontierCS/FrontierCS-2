// Trusted source policy for the editable package. Usage:
//
//	policy <trusted code dir> <candidate code dir>
//
// Everything is decided on the parsed syntax tree of every file in pir/;
// the evaluator separately checks the compiled package's import graph and
// that every file outside pir/ is byte-identical to the trusted tree. The
// evaluator always builds this tool from its own trusted copy.
package main

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Explicit allowlist: import path -> names that may follow the package name.
// A nil set allows every exported name (standard-library packages whose
// public API is memory-safe and has no shared mutable state, clock, file or
// network access). For the trusted packages of this repository every allowed
// name is listed; each was audited as described in the statement's "Safe
// API" section: no slice or raw pointer reaches assembly, no package state.
var allowedImports = map[string]map[string]bool{
	"bytes":           nil,
	"encoding/binary": nil,
	"math":            nil,
	"math/bits":       nil,
	"sort":            nil,
	"sync":            nil,
	"sync/atomic":     nil,
	// Only explicitly seeded generators: the package-level source is shared state.
	"math/rand": names("New NewSource Rand Source Source64"),
	// AES through fixed-size array pointers only.
	"easypir/safeaes": names("RoundKeys ExpandKey MMO Encrypt Xor16"),
	// Entry and key types and their pure-Go helpers; none of the PRF or cipher helpers.
	"example.com/util": names("DBEntry DBEntryLength DBEntrySize KeyByteLength PrfKey PrfKey128 " +
		"ZeroEntry DBEntryXor DBEntryXorFromRaw DBEntryFromSlice EntryIsEqual RandKey"),
	// Parameters, message encodings, the request generator and the typed reference PRF.
	"easypir/refwire": names("Params NoLabel FailureProbLog2 " +
		"EncodeRequest DecodeRequest EncodeResponse DecodeResponse " +
		"LongKey NewLongKey Eval SubKey SetOffset Generator NewGenerator LastQuery"),
}

func names(list string) map[string]bool {
	set := map[string]bool{}
	for _, name := range strings.Fields(list) {
		set[name] = true
	}
	return set
}

var allowedDirectives = map[string]bool{"//go:noinline": true, "//go:noescape": true}

func reject(file, reason string) {
	fmt.Fprintf(os.Stderr, "source policy rejected %s: %s\n", file, reason)
	os.Exit(2)
}

func main() {
	if len(os.Args) != 3 {
		reject("", "usage")
	}
	trusted := filepath.Join(os.Args[1], "pir")
	directory := filepath.Join(os.Args[2], "pir")
	entries, err := os.ReadDir(directory)
	if err != nil {
		reject("pir", "unreadable")
	}
	if _, err := os.Stat(trusted); err != nil {
		reject("pir", "trusted copy missing")
	}
	var scanned []string
	for _, entry := range entries {
		name := entry.Name()
		info, err := os.Lstat(filepath.Join(directory, name))
		if err != nil || !info.Mode().IsRegular() {
			reject(name, "only regular files are allowed in pir/")
		}
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			reject(name, "only non-test .go files may be added")
		}
		check(filepath.Join(directory, name), name)
		scanned = append(scanned, name)
	}
	sort.Strings(scanned)
	json.NewEncoder(os.Stdout).Encode(map[string]interface{}{"editable_go_files": scanned})
}

func check(path, name string) {
	source, err := os.ReadFile(path)
	if err != nil {
		reject(name, "unreadable")
	}
	positions := token.NewFileSet()
	file, err := parser.ParseFile(positions, name, source, parser.ParseComments)
	if err != nil {
		reject(name, "does not parse")
	}
	if file.Name.Name != "pir" {
		reject(name, "package must be pir")
	}
	for _, group := range file.Comments {
		for _, comment := range group.List {
			text := comment.Text
			body := strings.TrimLeft(strings.TrimPrefix(strings.TrimPrefix(text, "//"), "/*"), " \t")
			if strings.HasPrefix(body, "go:") && !(allowedDirectives[strings.TrimSpace(text)] && strings.HasPrefix(text, "//go:")) {
				reject(name, "compiler directive "+strings.TrimSpace(text))
			}
			if strings.HasPrefix(text, "//line ") || strings.HasPrefix(text, "/*line ") || strings.HasPrefix(body, "+build") {
				reject(name, "directive-like comment "+strings.TrimSpace(text))
			}
		}
	}
	// Local name of every import; dot and blank imports are not allowed.
	imports := map[string]string{}
	for _, spec := range file.Imports {
		importPath, err := strconv.Unquote(spec.Path.Value)
		if err != nil {
			reject(name, "malformed import")
		}
		if _, ok := allowedImports[importPath]; !ok {
			reject(name, "import "+importPath+" is not on the allowlist")
		}
		local := importPath[strings.LastIndex(importPath, "/")+1:]
		if spec.Name != nil {
			local = spec.Name.Name
		}
		if local == "." || local == "_" {
			reject(name, "dot and blank imports are not allowed")
		}
		if previous, ok := imports[local]; ok && previous != importPath {
			reject(name, "ambiguous import name "+local)
		}
		imports[local] = importPath
	}
	for _, declaration := range file.Decls {
		switch current := declaration.(type) {
		case *ast.GenDecl:
			if current.Tok == token.VAR {
				reject(name, "package-level variables are not allowed")
			}
		case *ast.FuncDecl:
			if current.Body == nil {
				reject(name, "functions without a body are not allowed")
			}
			if current.Recv == nil && current.Name.Name == "init" {
				reject(name, "init functions are not allowed")
			}
		}
	}
	ast.Inspect(file, func(node ast.Node) bool {
		selector, ok := node.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		owner, ok := selector.X.(*ast.Ident)
		if !ok {
			return true
		}
		importPath, imported := imports[owner.Name]
		if !imported {
			return true
		}
		if names := allowedImports[importPath]; names != nil && !names[selector.Sel.Name] {
			reject(name, importPath+"."+selector.Sel.Name+" is not allowed")
		}
		return true
	})
}
