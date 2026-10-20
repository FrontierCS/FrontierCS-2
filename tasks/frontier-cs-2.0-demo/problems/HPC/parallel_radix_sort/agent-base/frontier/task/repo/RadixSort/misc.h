#ifndef _S_MISC_INCLUDED
#define _S_MISC_INCLUDED



template <class E>
void swap_E(E *x, E* y) {
  E temp = *x;
  *x = *y;
  *y = temp;
};


// a function to extract "bits" bits starting at bit location "offset"
template <class E, class F>
  struct eBits {
    F _f;  int _mask;  intT _offset;
  eBits(int bits, intT offset, F f): _mask((1<<bits)-1),
      _offset(offset), _f(f) {}
    inline intT operator() (E p) {return (_mask&(_f(p))>>_offset);}
  };

template <class E, class F>
  void printBinary(E z, F f, int bits_count) {
  //int x = std::get<0>(z); //z.first();
  //F extract = eBits<E, F>(8, 0, f);
  int x = f(z);//extract(z);
  int bits[8];
  for(int i = 0; i < 8; i++) {
    bits[i] = x % 2;
    x /= 2;
  }
  for(int i = bits_count - 1; i >= 0; i--)
    printf("%d",bits[i]);
  return;
};

template <class E, class F, class G>
  void printArray(E * A, int start, int n, F f, G g){
  printf("=========\n");
  for(int i = 0; i < n; i++) {
    printf("A[%d] = %d\t", start + i, A[i]);
    printBinary(A[i], f, MAX_RADIX);
    printf(" ");
    printBinary(A[i], g, 7);
    printf("\n");
  }
  printf("\n");
  printf("=========\n");
};



#endif
