#ifndef _S_COMMONSORT_INCLUDED
#define _S_COMMONSORT_INCLUDED
#include "misc.h"
#include "radix_configs.h"

template< class _Type>
//gcc can't inline this funciton for some reson 
inline void _swap(_Type &a, _Type&b)  {
  _Type temp = b;
  b = a;
  a = temp;
}

template< class _Type, typename F>
  inline void insertionSortSimilarToSTLnoSelfAssignment( _Type* a, unsigned long a_size, F extract )
{

  for ( unsigned long i = 1; i < a_size; i++ )
    {
      if ( a[ i ] < a[ i - 1 ] )		// no need to do (j > 0) compare for the first iteration
	{
	  _Type currentElement = a[ i ];
	  auto compare = extract(currentElement); 
	  a[ i ] = a[ i - 1 ];
	  unsigned long j;
	  for ( j = i - 1; j > 0 && compare < extract(a[ j - 1 ]); j-- )
	    {
	      a[ j ] = a[ j - 1 ];
	    }
	  a[ j ] = currentElement;	// always necessary work/write
	}
      // Perform no work at all if the first comparison fails - i.e. never assign an element to itself!
    }
}


template< class _Type>
inline _Type extractElement(_Type e, unsigned const long shiftRightAmount){
  return (( e >> shiftRightAmount));
}

// Listing 3
template< class _Type, typename F>
  void _RadixSort_PowerOf2Radix_Serial_Full_Sort( _Type* a, long last, unsigned long shiftRightAmount, F extractBits, long * count_result = NULL, bool recurse = true)
{

  const unsigned long Log2ofPowerOfTwoRadix = MAX_RADIX;
  const unsigned long PowerOfTwoRadix = BUCKETS;
  const unsigned long numberOfBins = PowerOfTwoRadix;
  unsigned long count[ numberOfBins ];
  const unsigned long mask = (1<<Log2ofPowerOfTwoRadix)-1; 
  //long* count; 

  /* if(count_result){
     count = count_result; 
     }
     else{
     count = countStack; 
     }*/

  for( unsigned long i = 0; i < numberOfBins; i++ )
    count[ i ] = 0;
  for ( long _current = 0; _current <= last; _current++ ) // Scan the array and count the number of times each value appears
    {
      unsigned long digit = (unsigned long)( mask & (extractBits(a[ _current ]) >> shiftRightAmount) ) ; // extract the digit we are sorting based on
      count[ digit ]++;
    }

  if(count_result){
    for ( long _current = 0; _current < PowerOfTwoRadix; _current++ ){
      count_result[_current] = count[_current];
    
    }
  }
  

  long startOfBin[ numberOfBins ], endOfBin[ numberOfBins ], nextBin;
  startOfBin[ 0 ] = endOfBin[ 0 ] = nextBin = 0;
  for( unsigned long i = 1; i < numberOfBins; i++ )
    startOfBin[ i ] = endOfBin[ i ] = startOfBin[ i - 1 ] + count[ i - 1 ];
  
  long end = last-(long)count[PowerOfTwoRadix-1]; 
  for ( long _current = 0; _current <= end ; )
    {
      unsigned long digit;
      _Type tmp = a[ _current ];  // get the compiler to recognize that a register can be used for the loop instead of a[_current] memory location
      while ( true ) {
        digit = (unsigned long)(mask & ( extractBits(tmp)   >> shiftRightAmount ));   // extract the digit we are sorting based on
        if ( endOfBin[ digit ] == _current )
	  break;
	_swap( tmp, a[ endOfBin[ digit ] ] );
	endOfBin[ digit ]++;
      }
      a[ _current ] = tmp;
      endOfBin[ digit ]++;   // leave the element at its location and grow the bin
      _current++;  // advance the current pointer to the next element
      while( _current >= startOfBin[ nextBin ] && nextBin < numberOfBins )
	nextBin++;
      while( endOfBin[ nextBin - 1 ] == startOfBin[ nextBin ] && nextBin < numberOfBins )
	nextBin++;
      if ( _current < endOfBin[ nextBin - 1 ] )
	_current = endOfBin[ nextBin - 1 ];
    }


  if(recurse){
    if ( shiftRightAmount != 0 )   // end recursion when all the bits have been processes
      {
	if ( shiftRightAmount >= Log2ofPowerOfTwoRadix ) shiftRightAmount -= Log2ofPowerOfTwoRadix;
	else shiftRightAmount  = 0;

	parallel_for_1( int i = 0; i < numberOfBins; i++ )
	  {
	    long numberOfElements = endOfBin[ i ] - startOfBin[ i ];
	    if ( numberOfElements >= INSERTION_THRESHOLD )  // endOfBin actually points to one beyond the bin
              _RadixSort_PowerOf2Radix_Serial_Full_Sort( &a[ startOfBin[ i ]], numberOfElements - 1, shiftRightAmount, extractBits );
	    else if ( numberOfElements >= 2 )
	      insertionSortSimilarToSTLnoSelfAssignment( &a[ startOfBin[ i ]], numberOfElements, extractBits);
	  }
      }
  }

}

template <class E, class F, class bint>
  void radixtoPlaceSort(E* A, bint *counts,
                        long n, long buckets, F extract) {

  //printf("start %p count is %p -> %p [%p, %p] %d\n", A, counts, &counts[BUCKETS - 1], A, A+n,getWorkerId());
  bint currentHeadOffsets[BUCKETS];
  bint tailOffsets[BUCKETS];

  const unsigned long bitMask = extract._mask; 
  const unsigned long shiftRightAmount = extract._offset; 

  //printf("count init %p %d\n", A,getWorkerId());
  for (int i = 0; i < BUCKETS; i++) {
    counts[i] = 0;
  }


  //printf("extracting %p %d\n", A,getWorkerId());
  for (long j = 0; j < n; j++) {
    //bint k =extractElement( A[j] , bitMask, shiftRightAmount);
    int k = extractElement(A[j]);
    counts[k]++;
  }


  //printf(" tail offset count %p\n", A);
  bint s = 0;
  for (int i = 0; i < BUCKETS; i++) {
    currentHeadOffsets[i] = s;
    s += counts[i];
    tailOffsets[i] = s;
  }


  for (long i = 0; i < BUCKETS; i++) {
    while(currentHeadOffsets[i] < tailOffsets[i]) {
      E v = A[currentHeadOffsets[i]];
      //long bucketOfElement = extractElement(v, bitMask, shiftRightAmount);
      long bucketOfElement = extractElement(v);
      while(bucketOfElement != i) {
        E temp = v;
        v = A[currentHeadOffsets[bucketOfElement]];
        A[currentHeadOffsets[bucketOfElement]] = temp;
        currentHeadOffsets[bucketOfElement] ++;
        //bucketOfElement = extractElement(v, bitMask, shiftRightAmount);
        bucketOfElement = extractElement(v);
      }
      A[currentHeadOffsets[i] ++] = v;
    }
    
  }
}


template <class E, class F, class bint>
  void radixSortStable(E* A, bint *counts,
		       long n, F extract) {

  bint currentHeadOffsets[BUCKETS];

  for (int i = 0; i < BUCKETS; i++) {
    counts[i] = 0;
  }

  for (long j = 0; j < n; j++) {
    int k = extract(A[j]);
    counts[k]++;
  }

  bint s = 0;
  for (int i = 0; i < BUCKETS; i++) {
    currentHeadOffsets[i] = s;
    s += counts[i];
  }

  E* tempA = (E*) malloc(n * sizeof(E));
  for (long i = 0; i < n; i++) {
    long bucketOfElement = extract(A[i]);
    tempA[currentHeadOffsets[bucketOfElement]++] = A[i];
  }

  for(int i = 0; i < n; i++) {
    A[i] = tempA[i];
    //	  if(i > 0 && A[i].first == A[i-1].first && A[i].second < A[i-1].second) {
    //printf("problem A[%d] = %ul, while prev = %ul\n", i, A[i].first, A[i-1].first);
    //	  }
  }

  free(tempA);

}

//code from:
//http://www.drdobbs.com/architecture-and-design/algorithm-improvement-through-performanc/221600153?pgno=4
template< class _Type, class F>
  inline void _RadixSort_Unsigned_PowerOf2Radix_1( _Type* a, long *count_c, long n,  long buckets, F extract)
{
  _RadixSort_PowerOf2Radix_Serial_Full_Sort(a, n-1, (unsigned long ) extract._offset, extract._f, count_c, false); 
  return; 
}

template <class E, class F>
  void insertion_sort(E* A, long n, F f) {
  for(int i = 0; i < n; i++) {
    E x = A[i];
    int j = i - 1;
    while(j >= 0 && f(A[j]) > f(x)) {
      A[j+ 1] = A[j];
      j--;
    }
    A[j + 1] = x;
  }
  return;
}

template <class E, class F, class bint>
  void radixOneBittoPlaceSort(E* A, bint counts[BUCKETS],
			      long n, F extract) {


  long zeroPointer = 0;
  long onePointer = n;
  while(true){
    while(zeroPointer < onePointer && extract(A[zeroPointer]) == 0)
      zeroPointer ++;

    while(zeroPointer < onePointer && extract(A[onePointer - 1]) == 1)
      onePointer --;

    if(zeroPointer < onePointer) {
      swap_E(&A[zeroPointer], &A[onePointer - 1]);
    } else {
      break;
    }
  }

  counts[0] = zeroPointer;
  counts[1] = n - counts[0];

}


#endif
