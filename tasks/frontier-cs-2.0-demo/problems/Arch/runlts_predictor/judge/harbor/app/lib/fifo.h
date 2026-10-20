/*

Copyright (c) 2019, North Carolina State University
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. The names “North Carolina State University”, “NCSU” and any trade-name, personal name,
trademark, trade device, service mark, symbol, image, icon, or any abbreviation, contraction or
simulation thereof owned by North Carolina State University must not be used to endorse or promote products derived from this software without prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

// Author: Eric Rotenberg (ericro@ncsu.edu)


template <class T>
class fifo_t {
private:
    T *q;
    uint64_t size;
    uint64_t head;
    uint64_t tail;
    uint64_t length;

public:
    fifo_t(uint64_t size);
    ~fifo_t();
    bool empty();       // returns true if empty, false otherwise
    bool full();        // returns true if full, false otherwise
    T pop();        // pop and return head entry
    void push(T value); // push value at tail entry
  T peek();
    T peektail();       // examine value at tail entry
    T peekhead();       // examine value at head entry
};

template <class T>
fifo_t<T>::fifo_t(uint64_t size) {
   q = ((size > 0) ? (new T[size]) : ((T *)NULL));
   this->size = size;
   head = 0;
   tail = 0;
   length = 0;
}

template <class T>
fifo_t<T>::~fifo_t() {
}

template <class T>
bool fifo_t<T>::empty() {
   return(length == 0);
}

template <class T>
bool fifo_t<T>::full() {
   return(length == size);
}

// pop and return head entry
template <class T>
T fifo_t<T>::pop() {
   T ret = q[head];
   
   assert(length > 0);
   length--;
   head++;
   if (head == size)
      head = 0;

   return(ret);
}

// push value at tail entry
template <class T>
void fifo_t<T>::push(T value) {
   assert(length < size);
   length++;
   q[tail] = value;
   tail++;
   if (tail == size)
      tail = 0;
}

// examine value at tail entry
template <class T>
T fifo_t<T>::peektail() {
   // Tail actually points to next free entry for push, so examine previous entry w.r.t. tail.
   uint64_t prev = ((tail > 0) ? (tail - 1) : (size - 1));
   return(q[prev]);
}

// examine value at head entry
template <class T>
T fifo_t<T>::peekhead() {
   return(q[head]);
}
