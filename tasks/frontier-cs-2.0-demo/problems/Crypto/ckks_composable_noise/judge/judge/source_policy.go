// Judge-owned AST policy for the editable part of package estimator.
// It never implements estimator operations. The byte-identical fixed surface
// (statistics, test vectors, experiment mains, module files) is checked by
// fixed_surface.py before this program runs.
package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const module = "github.com/tuneinsight/ckks-noise-estimator"
const lattigo = "github.com/tuneinsight/lattigo/v6/"

var standard = map[string]bool{"fmt": true, "errors": true, "math": true, "math/big": true, "math/bits": true,
	"math/cmplx": true, "math/rand": true, "math/rand/v2": true, "container/heap": true, "sort": true,
	"slices": true, "cmp": true, "time": true, "runtime": true}

var lattigoPackages = map[string]bool{"core/rlwe": true, "schemes/ckks": true, "utils": true, "utils/bignum": true,
	"utils/sampling": true, "utils/structs": true, "circuits/ckks/bootstrapping": true, "circuits/ckks/dft": true,
	"circuits/ckks/lintrans": true, "circuits/ckks/mod1": true, "circuits/ckks/polynomial": true,
	"circuits/common/lintrans": true, "circuits/common/polynomial": true}

// Packages whose only permitted use is the listed function: everything else
// they export can schedule work that outlives the call that started it.
var restricted = map[string]string{"time": "Now", "runtime": "GC"}

// Files that are fixed byte for byte and therefore exempt from this policy.
var fixedFiles = map[string]bool{"estimator/stats.go": true, "estimator/utils.go": true}

// Declarations of an otherwise editable file that build the clear target.
var protected = map[string][]string{"estimator/dft.go": {"type:DFTMatrix", "*DFTMatrix.GenMatrices"}}

func fail(reason string) { fmt.Println("source policy violation:", reason); os.Exit(1) }

func normalized(n ast.Node) string {
	var b bytes.Buffer
	if printer.Fprint(&b, token.NewFileSet(), n) != nil {
		fail("unprintable declaration")
	}
	return b.String()
}

func allowedImport(path string) bool {
	if standard[path] || path == module {
		return true
	}
	if strings.HasPrefix(path, module+"/") {
		return !strings.HasPrefix(path, module+"/experiments")
	}
	if strings.HasPrefix(path, lattigo) {
		rest := strings.TrimPrefix(path, lattigo)
		return lattigoPackages[rest] || rest == "ring" || strings.HasPrefix(rest, "ring/")
	}
	return false
}

func declarations(f *ast.File) map[string]string {
	out := map[string]string{}
	for _, d := range f.Decls {
		switch x := d.(type) {
		case *ast.FuncDecl:
			key := x.Name.Name
			if x.Recv != nil {
				key = normalized(x.Recv.List[0].Type) + "." + key
			}
			out[key] = normalized(x)
		case *ast.GenDecl:
			if x.Tok == token.TYPE {
				for _, spec := range x.Specs {
					out["type:"+spec.(*ast.TypeSpec).Name.Name] = normalized(spec)
				}
			}
		}
	}
	return out
}

func check(path, rel string) *ast.File {
	raw, err := os.ReadFile(path)
	if err != nil {
		fail("unreadable source " + rel)
	}
	if bytes.Contains(raw, []byte("//go:")) || bytes.Contains(raw, []byte("//export")) || bytes.Contains(raw, []byte("+build")) {
		fail("compiler directive in " + rel)
	}
	f, err := parser.ParseFile(token.NewFileSet(), path, raw, 0)
	if err != nil {
		fail("unparsable source " + rel)
	}
	if filepath.Dir(rel) == "estimator" && f.Name.Name != "estimator" {
		fail("package name changed in " + rel)
	}
	if f.Name.Name == "main" {
		fail("additional program in " + rel)
	}
	names := map[string]string{}
	for _, im := range f.Imports {
		p, e := strconv.Unquote(im.Path.Value)
		if e != nil || !allowedImport(p) {
			fail("import not permitted in " + rel + ": " + im.Path.Value)
		}
		if im.Name != nil && (im.Name.Name == "_" || im.Name.Name == ".") {
			fail("blank or dot import in " + rel)
		}
		if only, ok := restricted[p]; ok {
			local := p
			if im.Name != nil {
				local = im.Name.Name
			}
			names[local] = only
		}
	}
	for _, d := range f.Decls {
		switch x := d.(type) {
		case *ast.FuncDecl:
			if x.Name.Name == "init" && x.Recv == nil {
				fail("package initialization function in " + rel)
			}
		case *ast.GenDecl:
			if x.Tok == token.VAR {
				ast.Inspect(x, func(n ast.Node) bool {
					switch n.(type) {
					case *ast.CallExpr, *ast.FuncLit:
						fail("package-level initializer performs work in " + rel)
					}
					return true
				})
			}
		}
	}
	ast.Inspect(f, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.GoStmt:
			fail("go statement in " + rel)
		case *ast.SelectStmt, *ast.SendStmt, *ast.ChanType:
			fail("channel operation in " + rel)
		case *ast.SelectorExpr:
			if id, ok := x.X.(*ast.Ident); ok && id.Obj == nil {
				if only, restrictedPackage := names[id.Name]; restrictedPackage && x.Sel.Name != only {
					fail("only " + id.Name + "." + only + " is permitted, in " + rel)
				}
			}
		}
		return true
	})
	return f
}

func main() {
	if len(os.Args) != 3 {
		fail("usage")
	}
	base, tree := os.Args[1], os.Args[2]
	root := filepath.Join(tree, "estimator")
	err := filepath.Walk(root, func(p string, info os.FileInfo, e error) error {
		if e != nil {
			fail("unreadable tree")
		}
		rel, _ := filepath.Rel(tree, p)
		rel = filepath.ToSlash(rel)
		if info.IsDir() {
			if rel == "estimator/experiments" {
				return filepath.SkipDir
			}
			return nil
		}
		if info.Mode()&os.ModeType != 0 {
			fail("special file " + rel)
		}
		if !strings.HasSuffix(rel, ".go") || fixedFiles[rel] {
			return nil
		}
		if strings.HasSuffix(rel, "_test.go") {
			fail("test file " + rel)
		}
		f := check(p, rel)
		if keys, ok := protected[rel]; ok {
			want, err := parser.ParseFile(token.NewFileSet(), filepath.Join(base, rel), nil, 0)
			if err != nil {
				fail("pristine source unreadable")
			}
			a, b := declarations(want), declarations(f)
			for _, key := range keys {
				if a[key] == "" || a[key] != b[key] {
					fail("protected declaration changed: " + key)
				}
			}
		}
		return nil
	})
	if err != nil {
		fail("walk")
	}
	for rel := range protected {
		if _, err := os.Stat(filepath.Join(tree, rel)); err != nil {
			fail("protected file removed: " + rel)
		}
	}
}
