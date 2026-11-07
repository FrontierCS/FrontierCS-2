//          Copyright Malte Skarupke 2016.
// Distributed under the Boost Software License, Version 1.0.
//    (See http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cstdint>
#include <algorithm>
#include <type_traits>
#include <tuple>
#include <utility>
#include <iostream>
#include "radix_configs.h"
#include "types.h"

static constexpr size_t invalidMetaDataShift = 1000;
//template <class sizeT>
struct  metaData  {
  bool recurse = false;
  bool * done = nullptr;
  bool * topLevel = nullptr;
  long * counts = nullptr;
  size_t rawShift = 1000;

  metaData(size_t rawShift_  = 1000, bool init = false, bool recurse_ = true, long* counts_ = nullptr ) {
    rawShift = rawShift_;
    if(init){
      done = new bool();
      topLevel = new bool();
      *done = false;
      *topLevel = true;
    }
    recurse = recurse_;
    counts = counts_;

  }


  // only top recursion layer can change this
  void setDone(bool val){
    if(topLevel && *topLevel && done){

      *done = val;
    }
  }

  void topLevelDone(){
    if(topLevel && *topLevel){
      *topLevel = false;
    }
  }

  void freeUnderlying(){
    if(topLevel) free(topLevel);
    if(done) free(done);
  }
};




namespace detail
{


  template<typename>
  struct RadixSorter;


  template<typename...>
  struct nested_void
  {
    using type = void;
  };

  template<typename... Args>
  using void_t = typename nested_void<Args...>::type;

  template<typename T>
  struct has_subscript_operator_impl
  {
    template<typename U, typename = decltype(std::declval<U>()[0])>
    static std::true_type test(int);
    template<typename>
    static std::false_type test(...);

    using type = decltype(test<T>(0));
  };

  template<typename T>
  using has_subscript_operator = typename has_subscript_operator_impl<T>::type;



  template<typename T>
  size_t radix_sort_pass_count = RadixSorter<T>::pass_count;

  template<typename It, typename Func>
  inline void unroll_loop_four_times(It begin, size_t iteration_count, Func && to_call)
  {
    size_t loop_count = iteration_count / 4;
    size_t remainder_count = iteration_count - loop_count * 4;
    for (; loop_count > 0; --loop_count)
      {
	to_call(begin);
	++begin;
	to_call(begin);
	++begin;
	to_call(begin);
	++begin;
	to_call(begin);
	++begin;
	/*to_call(begin);
	  ++begin;
	  to_call(begin);
	  ++begin;
	  to_call(begin);
	  ++begin;
	  to_call(begin);
	  ++begin;*/
      }
    switch(remainder_count)
      {
	/*case 7:
	  to_call(begin);
	  ++begin;
	  case 6:
	  to_call(begin);
	  ++begin;
	  case 5:
	  to_call(begin);
	  ++begin;
	  case 4:
	  to_call(begin);
	  ++begin;*/
      case 3:
	to_call(begin);
	++begin;
      case 2:
	to_call(begin);
	++begin;
      case 1:
	to_call(begin);
      }
  }

  template<typename It, typename F>
  inline It custom_std_partition(It begin, It end, F && func)
  {

    for (;; ++begin)
      {
	if (begin == end)
	  return end;
	if (!func(*begin))
	  break;
      }
    It it = begin;
    for(++it; it != end; ++it)
      {
	if (!func(*it))
	  continue;

	std::iter_swap(begin, it);
	++begin;
      }
    return begin;
  }

  struct PartitionInfo
  {
    PartitionInfo()
      : count(0)
    {
    }

    union
    {
      size_t count;
      size_t offset;
    };
    size_t next_offset;
  };



  template<typename T>
  struct SubKey;
  template<size_t Size>
  struct SizedSubKey
  {
    template<typename T>
    static auto sub_key(T && value, void *)
    {
      return to_unsigned_or_bool(value);
    }

    typedef SubKey<void> next;

    using sub_key_type = typename UnsignedForSize<Size>::type;
  };
  template<typename T>
  struct SubKey<const T> : SubKey<T>
  {
  };
  template<typename T>
  struct SubKey<T &> : SubKey<T>
  {
  };
  template<typename T>
  struct SubKey<T &&> : SubKey<T>
  {
  };
  template<typename T>
  struct SubKey<const T &> : SubKey<T>
  {
  };
  template<typename T>
  struct SubKey<const T &&> : SubKey<T>
  {
  };
  template<typename T, typename Enable = void>
  struct FallbackSubKey
    : SubKey<decltype(to_radix_sort_key(std::declval<T>()))>
  {
    using base = SubKey<decltype(to_radix_sort_key(std::declval<T>()))>;

    template<typename U>
    static decltype(auto) sub_key(U && value, void * data)
    {
      return base::sub_key(to_radix_sort_key(value), data);
    }
  };
  template<typename T>
  struct FallbackSubKey<T, void_t<decltype(to_unsigned_or_bool(std::declval<T>()))>>
    : SubKey<decltype(to_unsigned_or_bool(std::declval<T>()))>
  {
  };
  template<typename T>
  struct SubKey : FallbackSubKey<T>
  {
  };
  template<>
  struct SubKey<bool>
  {
    template<typename T>
    static bool sub_key(T && value, void *)
    {
      return value;
    }

    typedef SubKey<void> next;

    using sub_key_type = bool;
  };
  template<>
  struct SubKey<void>;
  template<>
  struct SubKey<unsigned char> : SizedSubKey<sizeof(unsigned char)>
  {
  };
  template<>
  struct SubKey<unsigned short> : SizedSubKey<sizeof(unsigned short)>
  {
  };
  template<>
  struct SubKey<unsigned int> : SizedSubKey<sizeof(unsigned int)>
  {
  };
  template<>
  struct SubKey<unsigned long> : SizedSubKey<sizeof(unsigned long)>
  {
  };
  template<>
  struct SubKey<unsigned long long> : SizedSubKey<sizeof(unsigned long long)>
  {
  };
  template<typename T>
  struct SubKey<T *> : SizedSubKey<sizeof(T *)>
  {
  };
  template<typename F, typename S, typename Current>
  struct PairSecondSubKey : Current
  {
    static decltype(auto) sub_key(const std::pair<F, S> & value, void * sort_data)
    {
      return Current::sub_key(value.second, sort_data);
    }

    using next = typename std::conditional<std::is_same<SubKey<void>, typename Current::next>::value, SubKey<void>, PairSecondSubKey<F, S, typename Current::next>>::type;
  };
  template<typename F, typename S, typename Current>
  struct PairFirstSubKey : Current
  {
    static decltype(auto) sub_key(const std::pair<F, S> & value, void * sort_data)
    {
      return Current::sub_key(value.first, sort_data);
    }

    using next = typename std::conditional<std::is_same<SubKey<void>, typename Current::next>::value, PairSecondSubKey<F, S, SubKey<S>>, PairFirstSubKey<F, S, typename Current::next>>::type;
  };
  template<typename F, typename S>
  struct SubKey<std::pair<F, S>> : PairFirstSubKey<F, S, SubKey<F>>
  {
  };
  template<size_t Index, typename First, typename... More>
  struct TypeAt : TypeAt<Index - 1, More..., void>
  {
  };
  template<typename First, typename... More>
  struct TypeAt<0, First, More...>
  {
    typedef First type;
  };




  struct BaseListSortData
  {
    size_t current_index;
    size_t recursion_limit;
    void * next_sort_data;
  };
  template<typename It, typename ExtractKey>
  struct ListSortData : BaseListSortData
  {
    void (*next_sort)(It, It, std::ptrdiff_t, ExtractKey &, void *);
  };

  template<typename CurrentSubKey, typename T>
  struct ListElementSubKey : SubKey<typename std::decay<decltype(std::declval<T>()[0])>::type>
  {
    using base = SubKey<typename std::decay<decltype(std::declval<T>()[0])>::type>;

    using next = ListElementSubKey;

    template<typename U>
    static decltype(auto) sub_key(U && value, void * sort_data)
    {
      BaseListSortData * list_sort_data = static_cast<BaseListSortData *>(sort_data);
      const T & list = CurrentSubKey::sub_key(value, list_sort_data->next_sort_data);
      return base::sub_key(list[list_sort_data->current_index], list_sort_data->next_sort_data);
    }
  };

  template<typename T>
  struct ListSubKey
  {
    using next = SubKey<void>;

    using sub_key_type = T;

    static const T & sub_key(const T & value, void *)
    {
      return value;
    }
  };

  template<typename T>
  struct FallbackSubKey<T, typename std::enable_if<has_subscript_operator<T>::value>::type> : ListSubKey<T>
  {
  };

  template<typename It, typename ExtractKey>
  inline void StdSortFallback(It begin, It end, ExtractKey & extract_key)
  {
    std::sort(begin, end, [&](auto && l, auto && r){ return extract_key(l) < extract_key(r); });
  }

  template<std::ptrdiff_t StdSortThreshold, typename It, typename ExtractKey>
  inline bool StdSortIfLessThanThreshold(It begin, It end, std::ptrdiff_t num_elements, ExtractKey & extract_key)
  {
    if (num_elements <= 1)
      return true;
    if (num_elements >= StdSortThreshold)
      return false;
    StdSortFallback(begin, end, extract_key);
    return true;
  }

  template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey, typename SubKeyType = typename CurrentSubKey::sub_key_type>
  struct InplaceSorter;

  template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey, size_t NumBytes, size_t Offset = 0>
  struct UnsignedInplaceSorter
  {
    static constexpr size_t ShiftAmount = (((NumBytes - 1) - Offset) * MAX_RADIX);

    static size_t getShift(size_t rawShift){
      size_t shift = (((NumBytes - 1) - Offset) * MAX_RADIX);

      if(rawShift != invalidMetaDataShift){
	if(Offset * MAX_RADIX > rawShift){
	  shift = 0;
	}else{
	  shift = rawShift - Offset * MAX_RADIX;
	}
      }

      return shift;
    }


    template<typename T>
    inline static radixCustomType index(T && elem, void * sort_data, size_t shiftAmount)
    {
      return (radixCustomType) (elem >> shiftAmount);
    }

    template<typename It, typename ExtractKey>
    static void sort(It begin, It end, std::ptrdiff_t num_elements, ExtractKey & extract_key, void (*next_sort)(It, It, std::ptrdiff_t, ExtractKey &, void *), void * sort_data, struct  metaData meta)
    {
      if (num_elements < AmericanFlagSortThreshold)
	american_flag_sort(begin, end, extract_key, next_sort, sort_data, meta);
      else
	ska_byte_sort(begin, end, extract_key, next_sort, sort_data, meta);
    }
    template<typename It, typename ExtractKey>
    static void american_flag_sort(It begin, It end, ExtractKey & extract_key, void (*next_sort)(It, It, std::ptrdiff_t, ExtractKey &, void *), void * sort_data, struct  metaData meta)
    {
      const unsigned long Log2ofPowerOfTwoRadix = MAX_RADIX;
      const unsigned long numberOfBins = BUCKETS;
      const unsigned long mask = (1<<Log2ofPowerOfTwoRadix)-1;
      PartitionInfo partitions[numberOfBins];

      size_t shift = getShift(meta.rawShift);
      for (It it = begin; it != end; ++it)
	{
	  ++partitions[mask & index(extract_key(*it), sort_data, shift)].count;
	}
      size_t total = 0;
      radixCustomType remaining_partitions[numberOfBins];
      int num_partitions = 0;
      for (int i = 0; i < numberOfBins; ++i)
	{
	  size_t count = partitions[i].count;
	  if(!meta.recurse  && meta.counts) {
	    meta.counts[i] = count;
	  }

	  if (!count)
	    continue;
	  partitions[i].offset = total;
	  total += count;
	  partitions[i].next_offset = total;
	  remaining_partitions[num_partitions] = i;
	  ++num_partitions;
	}
      if (num_partitions > 1)
	{
	  radixCustomType * current_block_ptr = remaining_partitions;
	  PartitionInfo * current_block = partitions + *current_block_ptr;
	  radixCustomType * last_block = remaining_partitions + num_partitions - 1;
	  It it = begin;
	  It block_end = begin + current_block->next_offset;
	  It last_element = end - 1;
	  for (;;)
	    {
	      PartitionInfo * block = partitions + (mask&index(extract_key(*it), sort_data, shift));
	      if (block == current_block)
		{
		  ++it;
		  if (it == last_element)
		    break;
		  else if (it == block_end)
		    {
		      for (;;)
			{
			  ++current_block_ptr;
			  if (current_block_ptr == last_block)
			    goto recurse;
			  current_block = partitions + *current_block_ptr;
			  if (current_block->offset != current_block->next_offset)
			    break;
			}

		      it = begin + current_block->offset;
		      block_end = begin + current_block->next_offset;
		    }
		}
	      else
		{
		  size_t offset = block->offset++;
		  std::iter_swap(it, begin + offset);
		}
	    }
	}

      if(num_partitions == 0) {
	meta.setDone(true);
      }
      else{
	meta.setDone(false);
      }


    recurse:
      if (meta.recurse && (shift != 0))
	{
	  size_t start_offset = 0;
	  It partition_begin = begin;
	  meta.setDone(true);
	  meta.topLevelDone();

	  for (radixCustomType * it = remaining_partitions, * end = remaining_partitions + num_partitions; it != end; ++it)
	    {
	      size_t end_offset = partitions[*it].next_offset;
	      It partition_end = begin + end_offset;
	      std::ptrdiff_t num_elements = end_offset - start_offset;
	      if (!StdSortIfLessThanThreshold<StdSortThreshold>(partition_begin, partition_end, num_elements, extract_key))
		{
		  UnsignedInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, NumBytes, Offset + 1>::sort(partition_begin, partition_end, num_elements, extract_key, next_sort, sort_data, meta);
		}
	      start_offset = end_offset;
	      partition_begin = partition_end;
	    }
	}
    }


    template<typename It, typename ExtractKey>
    static void _RadixSort_PowerOf2Radix_Serial_Full_Sort(It begin, It end,
							  ExtractKey & extract_key,
							  void (*next_sort)(It, It, std::ptrdiff_t, ExtractKey &, void *), void * sort_data, struct  metaData meta, bool topLevel = true
							  )
    {

      const unsigned long PowerOfTwoRadix = MAX_RADIX;
      const unsigned long numberOfBins = BUCKETS;
      const size_t shiftRightAmount = getShift(meta.rawShift);
      const unsigned long mask = (1<<PowerOfTwoRadix)-1;
      unsigned long count[ numberOfBins ];


      end -= 1;
      long _last = end - begin;

      for( unsigned long i = 0; i < numberOfBins; i++ )
	count[ i ] = 0;
      for (It it = begin; it <= end; ++it)// Scan the array and count the number of times each value appears
	{
	  count[ index(extract_key(*it), sort_data, shiftRightAmount) ]++;
	}

      //int num_partitions = 0;
      bool transferToCounts =!meta.recurse  && meta.counts;
      for (int i = 0; i < numberOfBins; ++i)
	{
	  long num = count[i];
	  if(transferToCounts) {
	    meta.counts[i] = num;
	  }
	  /*if(num){
	    num_partitions++;
	    }*/
	}

      /* if(num_partitions == 1){
	 goto recurse;
	 }*/

      long startOfBin[ numberOfBins ], endOfBin[ numberOfBins ], nextBin;
      startOfBin[ 0 ] = endOfBin[ 0 ] = nextBin = 0;
      for( unsigned long i = 1; i < numberOfBins; i++ )
	startOfBin[ i ] = endOfBin[ i ] = startOfBin[ i - 1 ] + count[ i - 1 ];
      for ( long _current = 0; _current <= _last-count[PowerOfTwoRadix-1]; )
	{
	  unsigned long digit;
	  It _currentIt = begin + _current;
	  auto tmp =  *_currentIt;  // get the compiler to recognize that a register can be used for the loop instead of a[_current] memory location
	  while ( true ) {
	    digit = (unsigned long)(index(extract_key(tmp), sort_data, shiftRightAmount));   // extract the digit we are sorting based on
	    if ( endOfBin[ digit ] == _current )
	      break;
	    auto tmp1 = tmp;
	    It dst = begin + endOfBin[ digit ];
	    tmp = *(dst);
	    *(dst) = tmp1;

	    endOfBin[ digit ]++;
	  }
	  *(begin + _current) = tmp;
	  endOfBin[ digit ]++;   // leave the element at its location and grow the bin
	  _current++;  // advance the current pointer to the next element
	  while( _current >= startOfBin[ nextBin ] && nextBin < numberOfBins )
	    nextBin++;
	  while( endOfBin[ nextBin - 1 ] == startOfBin[ nextBin ] && nextBin < numberOfBins )
	    nextBin++;
	  if ( _current < endOfBin[ nextBin - 1 ] )
	    _current = endOfBin[ nextBin - 1 ];
	}
      meta.setDone(false);
      return;

      /*recurse:
	if(num_partitions == 0) {
	meta.setDone(true);
	}
	else{
	meta.setDone(false);
	}*/
      /* if (meta.recurse && (shiftRightAmount !=0 || next_sort)){
	 meta.setDone(true);
	 meta.topLevelDone();
	 for( int i = 0; i < numberOfBins; i++ )
	 {
	 long num_elements = endOfBin[ i ] - startOfBin[ i ];
	 It partition_begin = begin + startOfBin[ i ];
	 It partition_end = partition_begin + num_elements; // endOfBin actually points to one beyond the bin
	 if (!StdSortIfLessThanThreshold<StdSortThreshold>(partition_begin, partition_end, num_elements, extract_key))
	 {
	 UnsignedInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, NumBytes, Offset + 1>::sort(partition_begin, partition_end, num_elements, extract_key, next_sort, sort_data, meta);
	 }
	 }
	 }*/
    }

    template<typename It, typename ExtractKey>
    static void ska_byte_sort(It begin, It end, ExtractKey & extract_key, void (*next_sort)(It, It, std::ptrdiff_t, ExtractKey &, void *), void * sort_data, struct  metaData meta, bool topLevel = true)
    {
      const unsigned long Log2ofPowerOfTwoRadix = MAX_RADIX;
      const unsigned long numberOfBins = BUCKETS;
      const unsigned long mask = (1<<Log2ofPowerOfTwoRadix)-1;
#if MAX_RADIX <= 8

      PartitionInfo partitions[numberOfBins];
#elif MAX_RADIX > 8  
      PartitionInfo * partitions =  new PartitionInfo[numberOfBins];
#endif

      const size_t shift = getShift(meta.rawShift);

      for (It it = begin; it != end; ++it)
	{
	  ++partitions[mask & index(extract_key(*it), sort_data, shift)].count;
	}


#if MAX_RADIX <= 8
      radixCustomType remaining_partitions[numberOfBins];
#elif MAX_RADIX > 8  
      radixCustomType * remaining_partitions =  new radixCustomType[numberOfBins];
#endif
      
      size_t total = 0;
      int num_partitions = 0;

      for (int i = 0; i < numberOfBins; ++i)
	{
	  size_t count = partitions[i].count;
	  if(!meta.recurse  && meta.counts) {
	    meta.counts[i] = count;
	  }

	  if (count)
	    {
	      partitions[i].offset = total;
	      total += count;
	      remaining_partitions[num_partitions] = i;
	      ++num_partitions;
	    }
	  partitions[i].next_offset = total;
	}
      for (radixCustomType * last_remaining = remaining_partitions + num_partitions, * end_partition = remaining_partitions + 1; last_remaining > end_partition;)
	{
	  last_remaining = custom_std_partition(remaining_partitions, last_remaining, [&](radixCustomType partition)
						{
						  size_t & begin_offset = partitions[partition].offset;
						  size_t & end_offset = partitions[partition].next_offset;
						  if (begin_offset == end_offset)
						    return false;

						  unroll_loop_four_times(begin + begin_offset, end_offset - begin_offset, [partitions = partitions, begin, &extract_key, sort_data, shift](It it)
	    {
	      radixCustomType this_partition = mask & index(extract_key(*it), sort_data, shift);
	      size_t offset = partitions[this_partition].offset++;
	      std::iter_swap(it, begin + offset);
	    });
						  return begin_offset != end_offset;
						});
	}

      if(num_partitions == 0) {
	meta.setDone(true);
      }
      else{
	meta.setDone(false);
      }

      if (meta.recurse && (shift != 0 || next_sort))
	{

	  meta.setDone(true);
	  meta.topLevelDone();

	  for (radixCustomType * it = remaining_partitions + num_partitions; it != remaining_partitions; --it)
	    {
	      radixCustomType partition = it[-1];
	      size_t start_offset = (partition == 0 ? 0 : partitions[partition - 1].next_offset);
	      size_t end_offset = partitions[partition].next_offset;
	      It partition_begin = begin + start_offset;
	      It partition_end = begin + end_offset;
	      std::ptrdiff_t num_elements = end_offset - start_offset;

	      if (!StdSortIfLessThanThreshold<StdSortThreshold>(partition_begin, partition_end, num_elements, extract_key))
		{
		  UnsignedInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, NumBytes, Offset + 1>::sort(partition_begin, partition_end, num_elements, extract_key, next_sort, sort_data, meta);
		}
	    }
	}

#if MAX_RADIX > 8 
      delete partitions; 
      delete remaining_partitions;
#endif
    }
  };

  template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey, size_t NumBytes>
  struct UnsignedInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, NumBytes, NumBytes>
  {
    template<typename It, typename ExtractKey>
    inline static void sort(It begin, It end, std::ptrdiff_t num_elements, ExtractKey & extract_key, void (*next_sort)(It, It, std::ptrdiff_t, ExtractKey &, void *), void * next_sort_data, struct  metaData meta)
    {

      next_sort(begin, end, num_elements, extract_key, next_sort_data);
    }
  };



  template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey, typename ListType>
  struct ListInplaceSorter
  {
    using ElementSubKey = ListElementSubKey<CurrentSubKey, ListType>;
    template<typename It, typename ExtractKey>
    static void sort(It begin, It end, ExtractKey & extract_key, ListSortData<It, ExtractKey> * sort_data)
    {
      size_t current_index = sort_data->current_index;
      void * next_sort_data = sort_data->next_sort_data;
      auto current_key = [&](auto && elem) -> decltype(auto)
	{
	  return CurrentSubKey::sub_key(extract_key(elem), next_sort_data);
	};
      auto element_key = [&](auto && elem) -> decltype(auto)
	{
	  return ElementSubKey::base::sub_key(elem, sort_data);
	};
      sort_data->current_index = current_index = CommonPrefix(begin, end, current_index, current_key, element_key);
      It end_of_shorter_ones = std::partition(begin, end, [&](auto && elem)
					      {
						return current_key(elem).size() <= current_index;
					      });
      std::ptrdiff_t num_shorter_ones = end_of_shorter_ones - begin;
      if (sort_data->next_sort && !StdSortIfLessThanThreshold<StdSortThreshold>(begin, end_of_shorter_ones, num_shorter_ones, extract_key))
	{
	  sort_data->next_sort(begin, end_of_shorter_ones, num_shorter_ones, extract_key, next_sort_data);
	}
      std::ptrdiff_t num_elements = end - end_of_shorter_ones;
      if (!StdSortIfLessThanThreshold<StdSortThreshold>(end_of_shorter_ones, end, num_elements, extract_key))
	{
	  void (*sort_next_element)(It, It, std::ptrdiff_t, ExtractKey &, void *) = static_cast<void (*)(It, It, std::ptrdiff_t, ExtractKey &, void *)>(&sort_from_recursion);
	  InplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, ElementSubKey>::sort(end_of_shorter_ones, end, num_elements, extract_key, sort_next_element, sort_data);
	}
    }

    template<typename It, typename ExtractKey>
    static void sort_from_recursion(It begin, It end, std::ptrdiff_t, ExtractKey & extract_key, void * next_sort_data)
    {
      ListSortData<It, ExtractKey> offset = *static_cast<ListSortData<It, ExtractKey> *>(next_sort_data);
      ++offset.current_index;
      --offset.recursion_limit;
      if (offset.recursion_limit == 0)
	{
	  StdSortFallback(begin, end, extract_key);
	}
      else
	{
	  sort(begin, end, extract_key, &offset);
	}
    }


    template<typename It, typename ExtractKey>
    static void sort(It begin, It end, std::ptrdiff_t, ExtractKey & extract_key, void (*next_sort)(It, It, std::ptrdiff_t, ExtractKey &, void *), void * next_sort_data)
    {
      ListSortData<It, ExtractKey> offset;
      offset.current_index = 0;
      offset.recursion_limit = 16;
      offset.next_sort = next_sort;
      offset.next_sort_data = next_sort_data;
      sort(begin, end, extract_key, &offset);
    }
  };



  template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey>
  struct InplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, uint8_t> : UnsignedInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, 1>
  {
  };
  template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey>
  struct InplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, uint16_t> : UnsignedInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, 2>
  {
  };
  template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey>
  struct InplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, uint32_t> : UnsignedInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, 4>
  {
  };
  template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey>
  struct InplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, uint64_t> : UnsignedInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, 8>
  {
  };
  template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey, typename SubKeyType, typename Enable = void>
  struct FallbackInplaceSorter;

  template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey, typename SubKeyType>
  struct InplaceSorter : FallbackInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, SubKeyType>
  {
  };

  template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey, typename SubKeyType>
  struct FallbackInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, SubKeyType, typename std::enable_if<has_subscript_operator<SubKeyType>::value>::type>
    : ListInplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey, SubKeyType>
  {
  };

  template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey>
  struct SortStarter;
  template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold>
  struct SortStarter<StdSortThreshold, AmericanFlagSortThreshold, SubKey<void>>
  {
    template<typename It, typename ExtractKey>
    static void sort(It, It, std::ptrdiff_t, ExtractKey &, void *)
    {
    }
  };

  template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename CurrentSubKey>
  struct SortStarter
  {
    template<typename It, typename ExtractKey>
    static void sort(It begin, It end, std::ptrdiff_t num_elements, ExtractKey & extract_key, void * next_sort_data = nullptr, struct  metaData meta = {})
    {
      if (StdSortIfLessThanThreshold<StdSortThreshold>(begin, end, num_elements, extract_key)) {
	meta.setDone(true);
	meta.topLevelDone();
	return;
      }


      void (*next_sort)(It, It, std::ptrdiff_t, ExtractKey &, void *) = static_cast<void (*)(It, It, std::ptrdiff_t, ExtractKey &, void *)>(&SortStarter<StdSortThreshold, AmericanFlagSortThreshold, typename CurrentSubKey::next>::sort);
      if (next_sort == static_cast<void (*)(It, It, std::ptrdiff_t, ExtractKey &, void *)>(&SortStarter<StdSortThreshold, AmericanFlagSortThreshold, SubKey<void>>::sort))
	next_sort = nullptr;
      InplaceSorter<StdSortThreshold, AmericanFlagSortThreshold, CurrentSubKey>::sort(begin, end, num_elements, extract_key, next_sort, next_sort_data, meta);
    }
  };

  template<std::ptrdiff_t StdSortThreshold, std::ptrdiff_t AmericanFlagSortThreshold, typename It, typename ExtractKey>
  void inplace_radix_sort(It begin, It end, ExtractKey & extract_key, struct  metaData meta)
  {
    using SubKey = SubKey<decltype(extract_key(*begin))>;
    SortStarter<StdSortThreshold, AmericanFlagSortThreshold, SubKey>::sort(begin, end, end - begin, extract_key, nullptr, meta);
  }

  struct IdentityFunctor
  {
    template<typename T>
    decltype(auto) operator()(T && i) const
    {
      return std::forward<T>(i);
    }
  };
}

template<typename It, typename ExtractKey>
static void ska_sort(It begin, It end, ExtractKey && extract_key, struct  metaData meta)
{
  detail::inplace_radix_sort<32, 1024>(begin, end, extract_key, meta);
}

template<typename It>
static void ska_sort(It begin, It end, struct  metaData meta)
{
  ska_sort(begin, end, detail::IdentityFunctor(), meta );
}

template<typename It, typename OutIt, typename ExtractKey>
bool ska_sort_copy(It begin, It end, OutIt buffer_begin, ExtractKey && key)
{
  std::ptrdiff_t num_elements = end - begin;
  if (num_elements < 128 || detail::radix_sort_pass_count<typename std::result_of<ExtractKey(decltype(*begin))>::type> >= 8)
    {
      ska_sort(begin, end, key);
      return false;
    }
  else
    return detail::RadixSorter<typename std::result_of<ExtractKey(decltype(*begin))>::type>::sort(begin, end, buffer_begin, key);
}
template<typename It, typename OutIt>
bool ska_sort_copy(It begin, It end, OutIt buffer_begin)
{
  return ska_sort_copy(begin, end, buffer_begin, detail::IdentityFunctor());
}
