package util

import (
	//"crypto/sha256"
	"hash/fnv"
	"math"

	// "fmt"
	//////////////////"io"
	//"crypto/aes"
	//"crypto/cipher"

	"log"

	"encoding/binary"
	rand "math/rand"

	uint256 "github.com/holiman/uint256"
	chacha20 "gitlab.com/yawning/chacha20.git"
	chacha20poly1305 "golang.org/x/crypto/chacha20poly1305"
)

const (
	DBEntrySize   = 16 // has to be a multiple of 8!!
	DBEntryLength = DBEntrySize / 8
	KeyByteLength = 16
)

type PrfKey256 [32]byte
type PrfNonce [12]byte
type PrfKey128 [16]byte
type block [16]byte
type DBEntry [DBEntryLength]uint64

type PrfKey PrfKey128

func RandKey256(rng *rand.Rand) PrfKey256 {
	var key [32]byte
	//rand.Read(key[:])
	binary.LittleEndian.PutUint64(key[0:8], rng.Uint64())
	binary.LittleEndian.PutUint64(key[8:16], rng.Uint64())
	binary.LittleEndian.PutUint64(key[16:24], rng.Uint64())
	binary.LittleEndian.PutUint64(key[24:32], rng.Uint64())
	return key
}

func RandKey128(rng *rand.Rand) PrfKey128 {
	var key [16]byte
	//rand.Read(key[:])
	binary.LittleEndian.PutUint64(key[0:8], rng.Uint64())
	binary.LittleEndian.PutUint64(key[8:16], rng.Uint64())
	return key
}

func RandKey(rng *rand.Rand) PrfKey {
	return PrfKey(RandKey128(rng))
}

func PRFEval(key *PrfKey, x uint64) uint64 {
	return PRFEval4((*PrfKey128)(key), x)
}

func DBEntryXor(dst *DBEntry, src *DBEntry) {
	for i := 0; i < DBEntryLength; i++ {
		(*dst)[i] ^= (*src)[i]
	}
}

func DBEntryXorFromRaw(dst *DBEntry, src []uint64) {
	for i := 0; i < DBEntryLength; i++ {
		(*dst)[i] ^= src[i]
	}
}

func EntryIsEqual(a *DBEntry, b *DBEntry) bool {
	for i := 0; i < DBEntryLength; i++ {
		if (*a)[i] != (*b)[i] {
			return false
		}
	}
	return true
}

func RandDBEntry(rng *rand.Rand) DBEntry {
	var entry DBEntry
	for i := 0; i < DBEntryLength; i++ {
		entry[i] = rng.Uint64()
	}
	return entry
}

func GenDBEntry(key uint64, id uint64) DBEntry {
	var entry DBEntry
	for i := 0; i < DBEntryLength; i++ {
		entry[i] = DefaultHash((key ^ id) + uint64(i))
	}
	return entry
}

func ZeroEntry() DBEntry {
	ret := DBEntry{}
	for i := 0; i < DBEntryLength; i++ {
		ret[i] = 0
	}
	return ret
}

func DBEntryFromSlice(s []uint64) DBEntry {
	var entry DBEntry
	for i := 0; i < DBEntryLength; i++ {
		entry[i] = s[i]
	}
	return entry
}

// return ChunkSize, SetSize
func GenParams(DBSize uint64) (uint64, uint64, uint64, uint64) {
	targetChunkSize := uint64(2 * math.Sqrt(float64(DBSize)))
	ChunkSize := uint64(1)
	for ChunkSize < targetChunkSize {
		ChunkSize *= 2
	}
	SetSize := uint64(math.Ceil(float64(DBSize) / float64(ChunkSize)))
	// round up to the next mulitple of 4
	SetSize = (SetSize + 3) / 4 * 4
	log.Printf("ChunkSize: %d, SetSize: %d\n", ChunkSize, SetSize)

	targetSBSize := uint64(2 * math.Sqrt(float64(SetSize)))
	SBSize := uint64(1)
	for SBSize < targetSBSize {
		SBSize *= 2
	}
	SBNum := uint64(math.Ceil(float64(SetSize) / float64(SBSize)))
	// round up to the next mulitple of 4
	SBNum = (SBNum + 3) / 4 * 4

	SetSize = SBNum * SBSize

	return ChunkSize, SetSize, SBSize, SBNum
}


type AesPrf struct {
	// block cipher.Block
	enc []uint32
}

func xor16(dst, a, b *byte)
func encryptAes128(xk *uint32, dst, src *byte)
func aes128MMO(xk *uint32, dst, src *byte)
func expandKeyAsm(key *byte, enc *uint32)

func NewCipher(key uint64) (*AesPrf, error) {
	k := make([]byte, 16)
	binary.LittleEndian.PutUint64(k, key)
	// The expanded key is 11 round keys of four words each.
	c := AesPrf{make([]uint32, 11*4)}
	expandKeyAsm(&k[0], &c.enc[0])
	// fmt.Println("NEW CIPHER")
	// fmt.Println(k)
	// fmt.Println(c.enc)
	return &c, nil
}

func (c *AesPrf) Encrypt(dst, src []byte) {
	if len(c.enc) < 11*4 || len(dst) < 16 || len(src) < 16 {
		panic("util: AesPrf.Encrypt needs a full key and 16-byte blocks")
	}
	encryptAes128(&c.enc[0], &dst[0], &src[0])
}

func DefaultHash(key uint64) uint64 {
	hash := fnv.New64a()
	b := make([]byte, 8)
	binary.LittleEndian.PutUint64(b, key)
	hash.Write(b)
	return hash.Sum64()
}

func nonSafePRFEval(key uint64, x uint64) uint64 {
	return DefaultHash(key ^ x)
}

func PRFEval1(key *uint256.Int, x uint64, nonce []byte, y []byte) uint64 {
	aead, _ := chacha20poly1305.New(key.Bytes())
	//nonce := make([]byte, chacha20poly1305.NonceSize)
	//y := make([]byte, 8)
	binary.LittleEndian.PutUint64(y, x)
	ciphertext := aead.Seal(nil, nonce, y, nil)
	return binary.LittleEndian.Uint64(ciphertext)
}

func PRFEval2(key *PrfKey, x uint64) uint64 {
	var nonce PrfNonce
	c, _ := chacha20.New((*key)[:], nonce[:])
	src := make([]byte, 8)
	binary.LittleEndian.PutUint64(make([]byte, 8), x)
	dsc := make([]byte, 8)
	c.XORKeyStream(dsc, src)
	return binary.LittleEndian.Uint64(dsc)
}

func PRFEval3(key *PrfKey128, x uint64) uint64 {
	//hash := fnv.New128a()
	//hash.Write((*key)[:])
	//longKey := blake2b.Sum256((*key)[:])
	var key2 PrfKey
	copy(key2[0:16], key[0:16])
	//copy(key2[16:32], *key[:])
	return PRFEval2(&key2, x)
}

func PRFEval4(key *PrfKey128, x uint64) uint64 {
	var longKey = make([]uint32, 11*4)
	expandKeyAsm(&key[0], &longKey[0])
	var src = make([]byte, 16)
	var dsc = make([]byte, 16)
	binary.LittleEndian.PutUint64(src, x)
	aes128MMO(&longKey[0], &dsc[0], &src[0])
	return binary.LittleEndian.Uint64(dsc)
}

func PRFEvalWithLongKey(longKey []uint32, x uint64) uint64 {
	var src = make([]byte, 16)
	var dsc = make([]byte, 16)
	binary.LittleEndian.PutUint64(src, x)
	aes128MMO(&longKey[0], &dsc[0], &src[0])
	return binary.LittleEndian.Uint64(dsc)
}

func PRFEvalWithLongKeyAndTag(longKey []uint32, tag uint32, x uint64) uint64 {
	var src = make([]byte, 16)
	var dsc = make([]byte, 16)
	binary.LittleEndian.PutUint64(src, (uint64(tag)<<35)+x)
	aes128MMO(&longKey[0], &dsc[0], &src[0])
	return binary.LittleEndian.Uint64(dsc)
}

func PRFEvalWithLongKeyAndTag128Output(longKey []uint32, tag uint32, x uint64) [16]byte {
	var src = make([]byte, 16)
	var dsc = make([]byte, 16)
	binary.LittleEndian.PutUint64(src, (uint64(tag)<<35)+x)
	aes128MMO(&longKey[0], &dsc[0], &src[0])
	var ret [16]byte
	copy(ret[:], dsc)
	return ret
}

func GetLongKey(key *PrfKey128) []uint32 {
	var longKey = make([]uint32, 11*4)
	expandKeyAsm(&key[0], &longKey[0])
	return longKey
}

type PRSet struct {
	Key PrfKey
}

func (p *PRSet) Expand(SetSize uint64, ChunkSize uint64) []uint64 {
	expandedSet := make([]uint64, SetSize)
	for i := uint64(0); i < SetSize; i++ {
		tmp := PRFEval(&p.Key, i)
		offset := tmp & (ChunkSize - 1)
		//tmpPrf.Evaluate(output)
		//offset := binary.LittleEndian.Uint64(output) & (ChunkSize - 1)
		expandedSet[i] = i*ChunkSize + offset
		//expandedSet[i] = i*ChunkSize + (DefaultHash(p.Key^i) % ChunkSize)
	}
	return expandedSet
}

func (p *PRSet) MembTest(id uint64, SetSize uint64, ChunkSize uint64) bool {
	ChunkID := id / ChunkSize
	// ensure Chunk size is a power of 2
	ChunkOffset := id & (ChunkSize - 1)

	tmp := PRFEval(&p.Key, ChunkID)
	offset := tmp & (ChunkSize - 1)
	//tmpPrf.Evaluate(output)
	// offset := binary.LittleEndian.Uint64(output) & (ChunkSize - 1)
	return ChunkOffset == offset
	// return ChunkOffset == (DefaultHash(p.Key^ChunkID) % ChunkSize)
}

func MembTest2(key *PrfKey, chunkID uint64, offset uint64, ChunkSize uint64) bool {
	// ensure Chunk size is a power of 2
	return offset == (PRFEval(key, chunkID) & (ChunkSize - 1))
	// return offset == (DefaultHash(p.Key^ChunkID) % ChunkSize)
}

func PRFEvalWithTag(key *PrfKey, tag uint32, x uint64) uint64 {
	return PRFEval(key, (uint64(tag)<<35)+x)
}

type PRSetWithShortTag struct {
	Tag uint32
}

func (p *PRSetWithShortTag) Expand(key *PrfKey, SetSize uint64, ChunkSize uint64) []uint64 {
	expandedSet := make([]uint64, SetSize)
	for i := uint64(0); i < SetSize; i++ {
		tmp := PRFEvalWithTag(key, p.Tag, i)
		offset := tmp & (ChunkSize - 1)
		//tmpPrf.Evaluate(output)
		//offset := binary.LittleEndian.Uint64(output) & (ChunkSize - 1)
		expandedSet[i] = i*ChunkSize + offset
		//expandedSet[i] = i*ChunkSize + (DefaultHash(p.Key^i) % ChunkSize)
	}
	return expandedSet
}

func (p *PRSetWithShortTag) ExpandWithLongKey(longKey []uint32, SetSize uint64, ChunkSize uint64) []uint64 {
	expandedSet := make([]uint64, SetSize)
	for i := uint64(0); i < SetSize; i++ {
		tmp := PRFEvalWithLongKeyAndTag(longKey, p.Tag, i)
		offset := tmp & (ChunkSize - 1)
		//tmpPrf.Evaluate(output)
		//offset := binary.LittleEndian.Uint64(output) & (ChunkSize - 1)
		expandedSet[i] = i*ChunkSize + offset
		//expandedSet[i] = i*ChunkSize + (DefaultHash(p.Key^i) % ChunkSize)
	}
	return expandedSet
}

func MembTestWithTag(key *PrfKey, tag uint32, chunkID uint64, offset uint64, ChunkSize uint64) bool {
	// ensure Chunk size is a power of 2
	return offset == (PRFEvalWithTag(key, tag, chunkID) & (ChunkSize - 1))
	// return offset == (DefaultHash(p.Key^ChunkID) % ChunkSize)
}

func MembTestWithLongKeyAndTag(longKey []uint32, tag uint32, chunkID uint64, offset uint64, ChunkSize uint64) bool {
	// ensure Chunk size is a power of 2
	return offset == (PRFEvalWithLongKeyAndTag(longKey, tag, chunkID) & (ChunkSize - 1))
	// return offset == (DefaultHash(p.Key^ChunkID) % ChunkSize)
}

// PRF implementation copied from https://bitbucket.org/henrycg/riposte/src/linear/prf/prf.go
// Length of PRF seed (in bytes)
// const KEY_LENGTH = 16

// type Key [KEY_LENGTH]byte

// func NewPrf(key uint64) (Prf, error) {
// 	var p Prf
// 	var err error

// 	k := make([]byte, 16)
// 	binary.LittleEndian.PutUint64(k, key)
// 	//fmt.Println(k)

// 	p.block, err = aes.NewCipher(k[:])
// 	return p, err
// }

// func (p *Prf) Evaluate(to_encrypt []byte) {
// 	// IV is all zeros (we will never use
// 	// this key again)
// 	iv := make([]byte, aes.BlockSize)

// 	// We are making the [unsafe] assumption that all blocks
// 	// are the same length.
// 	//iv_integer := block_idx * uint64(len(to_encrypt))
// 	//binary.PutUvarint(iv, iv_integer)

// 	stream := cipher.NewCTR(p.block, iv)
// 	stream.XORKeyStream(to_encrypt, to_encrypt)
// }
