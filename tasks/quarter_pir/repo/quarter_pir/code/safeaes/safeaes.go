// Package safeaes is the trusted, byte-locked AES-128 primitive for the
// editable package. It is the only place where submission-reachable code
// meets assembly, and its exported API is bounds-safe by type: every
// argument is a pointer to a fixed-size array, so the assembly can never be
// handed a buffer shorter than it reads or writes. The raw assembly entry
// points are unexported and cannot be named from another package.
package safeaes

// RoundKeys is an expanded AES-128 key: 11 round keys of 16 bytes.
type RoundKeys [44]uint32

//go:noescape
func xor16(dst, a, b *byte)

//go:noescape
func encryptAes128(xk *uint32, dst, src *byte)

//go:noescape
func aes128MMO(xk *uint32, dst, src *byte)

//go:noescape
func expandKeyAsm(key *byte, enc *uint32)

// ExpandKey writes the 11 round keys of key into roundKeys.
func ExpandKey(key *[16]byte, roundKeys *RoundKeys) {
	expandKeyAsm(&key[0], &roundKeys[0])
}

// MMO computes dst = AES(src) xor src (Matyas-Meyer-Oseas), the block
// function of the protocol's PRF.
func MMO(roundKeys *RoundKeys, dst, src *[16]byte) {
	aes128MMO(&roundKeys[0], &dst[0], &src[0])
}

// Encrypt computes dst = AES(src).
func Encrypt(roundKeys *RoundKeys, dst, src *[16]byte) {
	encryptAes128(&roundKeys[0], &dst[0], &src[0])
}

// Xor16 computes dst = a xor b.
func Xor16(dst, a, b *[16]byte) {
	xor16(&dst[0], &a[0], &b[0])
}
