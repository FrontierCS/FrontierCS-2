// Package rawrpc is the trusted online transport: one unary gRPC method that
// carries an opaque request blob and returns an opaque response blob.
package rawrpc

import (
	"context"
	"errors"

	"google.golang.org/grpc"
	"google.golang.org/grpc/encoding"
)

const (
	codecName  = "qpir-raw"
	methodName = "/qpir.Raw/Query"
)

type codec struct{}

func (codec) Marshal(v interface{}) ([]byte, error) {
	blob, ok := v.(*[]byte)
	if !ok {
		return nil, errors.New("rawrpc: unsupported message")
	}
	return *blob, nil
}

func (codec) Unmarshal(data []byte, v interface{}) error {
	blob, ok := v.(*[]byte)
	if !ok {
		return errors.New("rawrpc: unsupported message")
	}
	*blob = append([]byte(nil), data...)
	return nil
}

func (codec) Name() string { return codecName }

func init() { encoding.RegisterCodec(codec{}) }

// Handler answers one opaque request.
type Handler interface {
	RawQuery(request []byte) []byte
}

func unary(server interface{}, ctx context.Context, decode func(interface{}) error, _ grpc.UnaryServerInterceptor) (interface{}, error) {
	var request []byte
	if err := decode(&request); err != nil {
		return nil, err
	}
	response := server.(Handler).RawQuery(request)
	return &response, nil
}

func Register(server *grpc.Server, handler Handler) {
	server.RegisterService(&grpc.ServiceDesc{
		ServiceName: "qpir.Raw",
		HandlerType: (*Handler)(nil),
		Methods:     []grpc.MethodDesc{{MethodName: "Query", Handler: unary}},
	}, handler)
}

func Call(ctx context.Context, connection *grpc.ClientConn, request []byte) ([]byte, error) {
	var response []byte
	err := connection.Invoke(ctx, methodName, &request, &response, grpc.CallContentSubtype(codecName))
	return response, err
}
