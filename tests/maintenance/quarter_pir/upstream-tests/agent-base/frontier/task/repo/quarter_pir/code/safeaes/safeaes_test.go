package safeaes

import (
	"bytes"
	"crypto/aes"
	"encoding/binary"
	"encoding/hex"
	"math/rand"
	"testing"

	"easypir/refwire"

	"example.com/util"
)

func unhex(t *testing.T, s string) (out [16]byte) {
	raw, err := hex.DecodeString(s)
	if err != nil || len(raw) != 16 {
		t.Fatal("bad vector")
	}
	copy(out[:], raw)
	return
}

// FIPS-197 Appendix C.1 and Appendix B.
func TestKnownAnswers(t *testing.T) {
	for _, vector := range [][3]string{
		{"000102030405060708090a0b0c0d0e0f", "00112233445566778899aabbccddeeff", "69c4e0d86a7b0430d8cdb78070b4c55a"},
		{"2b7e151628aed2a6abf7158809cf4f3c", "3243f6a8885a308d313198a2e0370734", "3925841d02dc09fbdc118597196a0b32"},
	} {
		key, plain, want := unhex(t, vector[0]), unhex(t, vector[1]), unhex(t, vector[2])
		var keys RoundKeys
		var got, mmo, expected [16]byte
		ExpandKey(&key, &keys)
		Encrypt(&keys, &got, &plain)
		if got != want {
			t.Fatalf("Encrypt = %x, want %x", got, want)
		}
		MMO(&keys, &mmo, &plain)
		Xor16(&expected, &want, &plain)
		if mmo != expected {
			t.Fatalf("MMO = %x, want %x", mmo, expected)
		}
	}
}

func TestAgreesWithStandardLibraryAndReferencePRF(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 2000; i++ {
		var key, block, got, want [16]byte
		rng.Read(key[:])
		rng.Read(block[:])
		var keys RoundKeys
		ExpandKey(&key, &keys)
		Encrypt(&keys, &got, &block)
		standard, _ := aes.NewCipher(key[:])
		standard.Encrypt(want[:], block[:])
		if got != want {
			t.Fatalf("Encrypt differs from crypto/aes for key %x", key)
		}
		// The reference PRF: MMO of the little-endian input.
		x := rng.Uint64()
		var input, output [16]byte
		binary.LittleEndian.PutUint64(input[:], x)
		MMO(&keys, &output, &input)
		long := util.GetLongKey((*util.PrfKey128)(&key))
		if !bytes.Equal(binary.LittleEndian.AppendUint32(nil, long[43]), binary.LittleEndian.AppendUint32(nil, keys[43])) {
			t.Fatal("ExpandKey differs from util.GetLongKey")
		}
		if binary.LittleEndian.Uint64(output[:]) != util.PRFEvalWithLongKey(long, x) {
			t.Fatal("MMO differs from util.PRFEvalWithLongKey")
		}
		if binary.LittleEndian.Uint64(output[:]) != refwire.Eval(refwire.NewLongKey(&key), x) {
			t.Fatal("MMO differs from refwire.Eval")
		}
		tag := uint32(rng.Uint64())
		binary.LittleEndian.PutUint64(input[:], (uint64(tag)<<35)+x%1024)
		MMO(&keys, &output, &input)
		if output != refwire.SubKey(refwire.NewLongKey(&key), tag, x%1024) {
			t.Fatal("MMO differs from refwire.SubKey")
		}
	}
}

// The API leaves the buffers next to its arguments untouched.
func TestWritesStayInsideTheirArrays(t *testing.T) {
	var arena struct {
		before [16]byte
		keys   RoundKeys
		middle [16]byte
		block  [16]byte
		after  [16]byte
	}
	key := [16]byte{1, 2, 3}
	ExpandKey(&key, &arena.keys)
	MMO(&arena.keys, &arena.block, &key)
	Encrypt(&arena.keys, &arena.block, &key)
	Xor16(&arena.block, &key, &key)
	zero := [16]byte{}
	if arena.before != zero || arena.middle != zero || arena.after != zero {
		t.Fatal("write outside the destination array")
	}
}
