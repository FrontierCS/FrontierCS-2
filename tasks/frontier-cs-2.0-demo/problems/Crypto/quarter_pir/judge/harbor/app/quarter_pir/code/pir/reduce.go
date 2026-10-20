package pir

import "easypir/refwire"

// ReduceRequest maps the reference client's request for one query to the
// bytes this implementation sends instead. It runs in its own process and
// receives nothing but the public parameters and the reference request, so
// whatever the server sees is a function of the reference server view.
//
// The reference sends its own request unchanged.
func ReduceRequest(p refwire.Params, referenceRequest []byte) []byte {
	return referenceRequest
}
