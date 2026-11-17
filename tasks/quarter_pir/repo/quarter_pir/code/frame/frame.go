// Package frame is the trusted control-channel framing shared by the
// harness processes: one tag byte, a little-endian uint32 length, a payload.
package frame

import (
	"bufio"
	"encoding/binary"
	"errors"
	"io"
)

func Write(w *bufio.Writer, tag byte, payload []byte) error {
	var header [5]byte
	header[0] = tag
	binary.LittleEndian.PutUint32(header[1:], uint32(len(payload)))
	if _, err := w.Write(header[:]); err != nil {
		return err
	}
	if _, err := w.Write(payload); err != nil {
		return err
	}
	return w.Flush()
}

// Read returns the next frame. A nil buffer allocates; otherwise the payload
// is read into buffer, which must be large enough.
func Read(r *bufio.Reader, limit int, buffer []byte) (byte, []byte, error) {
	var header [5]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return 0, nil, err
	}
	size := int(binary.LittleEndian.Uint32(header[1:]))
	if size > limit {
		return 0, nil, errors.New("frame: payload exceeds limit")
	}
	if buffer == nil || len(buffer) < size {
		buffer = make([]byte, size)
	}
	if _, err := io.ReadFull(r, buffer[:size]); err != nil {
		return 0, nil, err
	}
	return header[0], buffer[:size], nil
}
