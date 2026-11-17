#ifndef _CSR_H_
#define _CSR_H_

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <unordered_set>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include "CSC.h"
//#include "CSR_FlengthCluster.h"
#include "Triple.h"

#include <tbb/scalable_allocator.h>

#include <random>
#include "utility.h"

using namespace std;

template <class IT, class NT>
class CSR
{ 
public:
    CSR():nnz(0), rows(0), cols(0),zerobased(true),is_sorted(false) {}
	CSR(IT mynnz, IT m, IT n):nnz(mynnz),rows(m),cols(n),zerobased(true),is_sorted(false)
	{
        // Constructing empty Csc objects (size = 0) are allowed (why wouldn't they?).
        assert(rows != 0);
        rowptr = my_malloc<IT>(rows + 1);
		if(nnz > 0) {
            colids = my_malloc<IT>(nnz);
            values = my_malloc<NT>(nnz);
        }
	}
    
    CSR (graph & G);
    CSR (string filename);
    CSR (Triple<IT,NT> * triples, IT mynnz, IT m, IT n); // COO -> CSR conversion
//    CSR (const CSR_FlengthCluster<IT,NT> & csr_flength_cluster);   // CSR_FlengthCluster -> CSR conversion
    CSR (const CSC<IT,NT> & csc);   // CSC -> CSR conversion
    CSR (const CSR<IT,NT> & rhs);	// copy constructor
    CSR (const CSC<IT,NT> & csc, const bool transpose);
	  CSR<IT,NT> & operator=(const CSR<IT,NT> & rhs);	// assignment operator
    bool operator==(const CSR<IT,NT> & rhs); // ridefinizione ==

    NT overlap_coefficient_similarity(int a, int b);  // calculate overlap-coefficient similarity score between row a and b
    NT calculate_average_oc_similarity();  // calculate average overlap-coefficient similarity score of the matrix

    NT jaccard_similarity(int a, int b);  // calculate jaccard similarity score between row a and b
    NT clustering_factor(int a, int b);  // calculate jaccard similarity score between row a and b
    IT common_elements(int a, int b);
    NT calculate_average_similarity();  // calculate average jaccard similarity score of the matrix
    vector<NT> calculate_average_similarity(IT ngram);  // calculate n-gram average jaccard similarity score of the matrix

    void gather_unique_col_ids_in_cluster(unordered_set<IT> &unique_cols, IT st, IT nd);
    NT calculate_average_unique_cols_in_cluster(IT cluster_sz);
    NT jaccard_similarity(const unordered_set<IT> &a, const unordered_set<IT> &b);
    NT calculate_average_similarity_in_cluster(IT cluster_sz);
    std::tuple<NT, NT, NT> calculate_similarity_stats();

    NT get_load_imbalance_factor(const IT thread_num, const CSR<IT,NT> & B);
    pair<IT, NT> calculate_offdiagonal_nonzero_count(int block_sz);
    pair<IT, NT> calculate_bandwidth();
    pair<IT, NT> calculate_profile();
    NT calculate_average_row_width();
    double calculate_size();

    void print_jaccard_similarity(const string& filename);  // print jaccard similarity score between consecutive rows
    void shuffleIds(); // Randomly permutating column indices
    void sortIds(); // Permutating column indices in ascending order
    void sortValues(); // Permutating value indices in descending order
    void print_rows(IT start, IT end);  // Print CSR rows
    void print_to_file(const string &filename);   // Print CSR to file
    void print_topK(const string &filename, IT topK, NT value_th);    // Print CSR to file
    void print_topK(const string &filename, IT topK);                 // Print CSR to file
    map<IT, map<IT, NT>> get_rowmap();  // used in reordering
    void load_reorderedCSR(map<IT, map<IT, NT>> &csr, vector<IT> &reordered_rows, int panel_size, map<IT, map<IT, NT>> &spp3);
    void loadSplittedMatrix(vector<pair<IT, map<IT, NT>>> &csr, vector<IT> &reordered_rows, int panel_size);
    
    void make_empty()
    {
        if(nnz > 0) {
            my_free<IT>(colids);
            my_free<NT>(values);
            nnz = 0;
        }
        if(rows > 0) {
            my_free<IT>(rowptr);
            rows = 0;
        }
        cols = 0;	
    }
    
    ~CSR()
	{
        make_empty();
	}
    bool ConvertOneBased()
    {
        if(!zerobased)	// already one-based
            return false; 
        transform(rowptr, rowptr + rows + 1, rowptr, bind2nd(plus<IT>(), static_cast<IT>(1)));
        transform(colids, colids + nnz, colids, bind2nd(plus<IT>(), static_cast<IT>(1)));
        zerobased = false;
        return true;
    }
    bool ConvertZeroBased()
    {
        if (zerobased)
            return true;
        transform(rowptr, rowptr + rows + 1, rowptr, bind2nd(plus<IT>(), static_cast<IT>(-1)));
        transform(colids, colids + nnz, colids, bind2nd(plus<IT>(), static_cast<IT>(-1)));
        zerobased = true;
        return false;
    }
    bool isEmpty()
    {
        return ( nnz == 0 );
    }
    void Sorted();
    void SortedValues();
    void ResetValues();
    
    IT rows;
    IT cols;
    IT nnz; // number of nonzeros
    
    IT *rowptr;
    IT *colids;
    NT *values;
    bool zerobased;
    bool is_sorted;       // sort by ids
    bool is_value_sorted; // sort by values
};

//! calculate size in Bytes
template <class IT, class NT>
double CSR<IT,NT>::calculate_size() {
  size_t total = sizeof(CSR<IT, NT>);  // size of the class instance itself
  total += ((rows + 1) * sizeof(IT));             // rowptr
  total += (nnz * sizeof(IT));                    // colids
  total += (nnz * sizeof(NT));                    // values
  // Convert bytes to gigabytes
//  double size_in_gb = static_cast<double>(total) / (1024 * 1024 * 1024);
//  cout << "Size of CSR matrix: " << size_in_gb << " GB" << endl;
//  return total;
  return static_cast<double>(total);
}

// copy constructor
template <class IT, class NT>
CSR<IT,NT>::CSR (const CSR<IT,NT> & rhs): nnz(rhs.nnz), rows(rhs.rows), cols(rhs.cols),zerobased(rhs.zerobased),is_sorted(rhs.is_sorted), is_value_sorted(rhs.is_value_sorted)
{
	if(nnz > 0)
	{
        values = my_malloc<NT>(nnz);
        colids = my_malloc<IT>(nnz);
        copy(rhs.values, rhs.values+nnz, values);
        copy(rhs.colids, rhs.colids+nnz, colids);
	}
	if ( rows > 0)
	{
        rowptr = my_malloc<IT>(rows + 1);
        copy(rhs.rowptr, rhs.rowptr+rows+1, rowptr);
	}
}

template <class IT, class NT>
CSR<IT,NT> & CSR<IT,NT>::operator= (const CSR<IT,NT> & rhs)
{
	if(this != &rhs)		
	{
		if(nnz > 0)	// if the existing object is not empty
		{
            my_free<IT>(colids);
            my_free<NT>(values);
		}
		if(rows > 0)
		{
            my_free<IT>(rowptr);
		}

		nnz	= rhs.nnz;
		rows = rhs.rows;
		cols = rhs.cols;
		zerobased = rhs.zerobased;
    is_sorted = rhs.is_sorted;
    is_value_sorted = rhs.is_value_sorted;
		if(rhs.nnz > 0)	// if the copied object is not empty
		{
            values = my_malloc<NT>(nnz);
            colids = my_malloc<IT>(nnz);
            copy(rhs.values, rhs.values+nnz, values);
            copy(rhs.colids, rhs.colids+nnz, colids);
		}
		if(rhs.cols > 0)
		{
            rowptr = my_malloc<IT>(rows + 1);
            copy(rhs.rowptr, rhs.rowptr+rows+1, rowptr);
		}
	}
	return *this;
}

//! Construct a CSR object from a COO
template <class IT, class NT>
CSR<IT,NT>::CSR(Triple<IT,NT> * triples, IT mynnz, IT m, IT n):nnz(mynnz),rows(m),cols(n),zerobased(true),is_sorted(true),is_value_sorted(false)
{
  rowptr = my_malloc<IT>(rows + 1);
  colids = my_malloc<IT>(nnz);
  values = my_malloc<NT>(nnz);

  vector< pair<IT,NT> > tosort (nnz);

  IT *work = my_malloc<IT>(rows);
  std::fill(work, work+rows, (IT) 0);

  for (IT k = 0 ; k < nnz ; ++k)
  {
    IT tmp =  triples[k].row;
    work [ tmp ]++ ;		// row counts (i.e, w holds the "row difference array")
  }

  if(nnz > 0)
  {
    rowptr[rows] = CumulativeSum (work, rows) ;		// cumulative sum of w
    copy(work, work+rows, rowptr);
    IT last;
    for (IT k = 0 ; k < nnz ; ++k)
    {
      tosort[ work[triples[k].row]++] = make_pair( triples[k].col, triples[k].val);
    }
#pragma omp parallel for
    for(IT i=0; i< rows; ++i)
    {
      sort(tosort.begin() + rowptr[i], tosort.begin() + rowptr[i+1]);

      typename vector<pair<IT,NT> >::iterator itr;	// iterator is a dependent name
      IT ind;
      for(itr = tosort.begin() + rowptr[i], ind = rowptr[i]; itr != tosort.begin() + rowptr[i+1]; ++itr, ++ind)
      {
        colids[ind] = itr->first;
        values[ind] = itr->second;
      }
    }
  }
  my_free<IT>(work);
}

//! Construct a CSR object from a CSC
//! Accepts only zero based CSC inputs
template <class IT, class NT>
CSR<IT,NT>::CSR(const CSC<IT,NT> & csc):nnz(csc.nnz), rows(csc.rows), cols(csc.cols),zerobased(true),is_sorted(true),is_value_sorted(false)
{
    rowptr = my_malloc<IT>(rows + 1);
    colids = my_malloc<IT>(nnz);
    values = my_malloc<NT>(nnz);

    IT *work = my_malloc<IT>(rows);
    std::fill(work, work+rows, (IT) 0); // initilized to zero
   
    	for (IT k = 0 ; k < nnz ; ++k)
    	{
        	IT tmp =  csc.rowids[k];
        	work [ tmp ]++ ;		// row counts (i.e, w holds the "row difference array")
	}

	if(nnz > 0)
	{
		rowptr[rows] = CumulativeSum (work, rows);		// cumulative sum of w
       	 	copy(work, work+rows, rowptr);

		IT last;
        	for (IT i = 0; i < cols; ++i) 
        	{
       	     		for (IT j = csc.colptr[i]; j < csc.colptr[i+1] ; ++j)
            		{
				colids[ last = work[ csc.rowids[j] ]++ ]  = i ;
				values[last] = csc.values[j] ;
            		}
        	}
	}
    my_free<IT>(work);
}

template <class IT, class NT>
CSR<IT,NT>::CSR(const CSC<IT,NT> & csc, const bool transpose):nnz(csc.nnz), rows(csc.rows), cols(csc.cols),zerobased(true),is_sorted(true),is_value_sorted(false)
{
    if (!transpose) {
        rowptr = my_malloc<IT>(rows + 1);
        colids = my_malloc<IT>(nnz);
        values = my_malloc<NT>(nnz);

        IT *work = my_malloc<IT>(rows);
        std::fill(work, work+rows, (IT) 0); // initilized to zero
   
    	for (IT k = 0 ; k < nnz ; ++k)
            {
                IT tmp =  csc.rowids[k];
                work [ tmp ]++ ;		// row counts (i.e, w holds the "row difference array")
            }

        if(nnz > 0) 
            {
                rowptr[rows] = CumulativeSum (work, rows);		// cumulative sum of w
                copy(work, work+rows, rowptr);

                IT last;
                for (IT i = 0; i < cols; ++i) 
                    {
                        for (IT j = csc.colptr[i]; j < csc.colptr[i+1] ; ++j)
                            {
                                colids[ last = work[ csc.rowids[j] ]++ ]  = i ;
                                values[last] = csc.values[j] ;
                            }
                    }
            }
        my_free<IT>(work);
    }
    else {
        rows = csc.cols;
        cols = csc.rows;
        rowptr = my_malloc<IT>(rows + 1);
        colids = my_malloc<IT>(nnz);
        values = my_malloc<NT>(nnz);

        for (IT k = 0; k < rows + 1; ++k) {
            rowptr[k] = csc.colptr[k];
        }
        for (IT k = 0; k < nnz; ++k) {
            values[k] = csc.values[k];
            colids[k] = csc.rowids[k];
        }
    }
}

//todo: not sure if is_sorted is set correctly???
template <class IT, class NT>
CSR<IT,NT>::CSR(graph & G):nnz(G.m), rows(G.n), cols(G.n), zerobased(true),is_sorted(true),is_value_sorted(false)
{
	// graph is like a triples object
	// typedef struct {
        // LONG_T m;
        // LONG_T n;
        // // Arrays of size 'm' storing the edge information
        // // A directed edge 'e' (0 <= e < m) from start[e] to end[e]
        // // had an integer weight w[e] 
        // LONG_T* start;
        // LONG_T* end; 
	// WEIGHT_T* w;
	// } graph; 
	cout << "Graph nnz= " << G.m << " and n=" << G.n << endl;

	vector< Triple<IT,NT> > simpleG;
	vector< pair< pair<IT,IT>,NT> > currCol;
	currCol.push_back(make_pair(make_pair(G.start[0], G.end[0]), G.w[0]));
	for (IT k = 0 ; k < nnz-1 ; ++k) {
        if(G.start[k] != G.start[k+1] ) {
            std::sort(currCol.begin(), currCol.end());
            simpleG.push_back(Triple<IT,NT>(currCol[0].first.first, currCol[0].first.second, currCol[0].second));
            for(int i=0; i< currCol.size()-1; ++i) {
                if(currCol[i].first == currCol[i+1].first) {
                    simpleG.back().val += currCol[i+1].second;
                }
                else {	
                    simpleG.push_back(Triple<IT,NT>(currCol[i+1].first.first, currCol[i+1].first.second, currCol[i+1].second));
                }
            }
            vector< pair< pair<IT,IT>,NT> >().swap(currCol);
        }
		currCol.push_back(make_pair(make_pair(G.start[k+1], G.end[k+1]), G.w[k+1]));
    }
    
	// now do the last row
	sort(currCol.begin(), currCol.end());
    simpleG.push_back(Triple<IT,NT>(currCol[0].first.first, currCol[0].first.second, currCol[0].second));
    for(int i=0; i< currCol.size()-1; ++i) {
        if(currCol[i].first == currCol[i+1].first) {
            simpleG.back().val += currCol[i+1].second;
        }
		else {
            simpleG.push_back(Triple<IT,NT>(currCol[i+1].first.first, currCol[i+1].first.second, currCol[i+1].second));
        }
    }

	nnz = simpleG.size();
	cout << "[After duplicate merging] Graph nnz= " << nnz << " and n=" << G.n << endl;

    rowptr = my_malloc<IT>(rows + 1);
    colids = my_malloc<IT>(nnz);
    values = my_malloc<NT>(nnz);

    IT *work = my_malloc<IT>(rows);
    std::fill(work, work+rows, (IT) 0); // initilized to zero
    
    for (IT k = 0 ; k < nnz ; ++k) {
        IT tmp =  simpleG[k].row;
        work [ tmp ]++ ;		// col counts (i.e, w holds the "col difference array")
	}

	if(nnz > 0) {
        rowptr[rows] = CumulativeSum (work, rows) ;		// cumulative sum of w
        copy(work, work + rows, rowptr);
        
		IT last;
		for (IT k = 0 ; k < nnz ; ++k) {
            colids[ last = work[ simpleG[k].row ]++ ]  = simpleG[k].col ;
			values[last] = simpleG[k].val ;
        }
	}
    my_free<IT>(work);
}


// check if sorted within rows?
template <class IT, class NT>
void CSR<IT,NT>::Sorted()
{
	bool sorted = true;
	for(IT i=0; i< rows; ++i)
	{
		sorted &= my_is_sorted (colids + rowptr[i], colids + rowptr[i+1], std::less<IT>());
    }
}

// check if values are sorted within rows?
template <class IT, class NT>
void CSR<IT,NT>::SortedValues()
{
  bool sorted = true;
  for(IT i=0; i< rows; ++i)
  {
    sorted &= my_is_sorted (values + rowptr[i], values + rowptr[i+1], std::greater<NT>());
  }
}

template <class IT, class NT>
void CSR<IT,NT>::ResetValues()
{
  for(IT i=0; i< nnz; ++i)
  {
    values[i] = 1;
  }
}

template <class IT, class NT>
bool CSR<IT,NT>::operator==(const CSR<IT,NT> & rhs)
{
    bool same;
    if(nnz != rhs.nnz || rows  != rhs.rows || cols != rhs.cols) {
        printf("%d:%d, %d:%d, %d:%d\n", nnz, rhs.nnz, rows, rhs.rows, cols, rhs.cols);
        return false;
    }
//    cout << "zerobased: " << zerobased << ", rhs.zerobased: " << rhs.zerobased << endl;
    if (zerobased != rhs.zerobased) {
        IT *tmp_rowptr = my_malloc<IT>(rows + 1);
        IT *tmp_colids = my_malloc<IT>(nnz);
        if (!zerobased) {
            for (int i = 0; i < rows + 1; ++i) {
                tmp_rowptr[i] = rowptr[i] - 1;
            }
            for (int i = 0; i < nnz; ++i) {
                tmp_colids[i] = colids[i] - 1;
            }
            same = std::equal(tmp_rowptr, tmp_rowptr + rows + 1, rhs.rowptr); 
            same = same && std::equal(tmp_colids, tmp_colids + nnz, rhs.colids);
//            cout << "Same check 1: " << same << endl;
        }
        else if (!rhs.zerobased) {
            for (int i = 0; i < rows + 1; ++i) {
                tmp_rowptr[i] = rhs.rowptr[i] - 1;
            }
            for (int i = 0; i < nnz; ++i) {
                tmp_colids[i] = rhs.colids[i] - 1;
            }
            same = std::equal(tmp_rowptr, tmp_rowptr + rows + 1, rowptr); 
            same = same && std::equal(tmp_colids, tmp_colids + nnz, colids);
//            cout << "Same check 2: " << same << endl;
        }
        my_free<IT>(tmp_rowptr);
        my_free<IT>(tmp_colids);
    }
    else {
        same = std::equal(rowptr, rowptr+rows+1, rhs.rowptr);
//        cout << "Same check 3.1: " << same << endl;
        same = same && std::equal(colids, colids+nnz, rhs.colids);
//        cout << "Same check 3.2: " << same << endl;
//        cout << "Same check 3.3: " << std::equal(colids, colids+nnz, rhs.colids) << endl;
    }
    
    bool samebefore = same;
    ErrorTolerantEqual<NT> epsilonequal(EPSILON);
    same = same && std::equal(values, values+nnz, rhs.values, epsilonequal );
//    cout << "Same check 4: " << same << endl;
    if(samebefore && (!same)) {
//#ifdef DEBUG
        vector<NT> error(nnz);
        transform(values, values+nnz, rhs.values, error.begin(), absdiff<NT>());
        vector< pair<NT, NT> > error_original_pair(nnz);
        for(IT i=0; i < nnz; ++i)
            error_original_pair[i] = make_pair(error[i], values[i]);
        if(error_original_pair.size() > 10) { // otherwise would crush for small data
            partial_sort(error_original_pair.begin(), error_original_pair.begin()+10, error_original_pair.end(), greater< pair<NT,NT> >());
            cout << "Highest 10 different entries are: " << endl;
            for(IT i=0; i < 10; ++i)
                cout << "Diff: " << error_original_pair[i].first << " on " << error_original_pair[i].second << endl;
        }
        else {
            sort(error_original_pair.begin(), error_original_pair.end(), greater< pair<NT,NT> >());
            cout << "Highest different entries are: " << endl;
            for(typename vector< pair<NT, NT> >::iterator it=error_original_pair.begin(); it != error_original_pair.end(); ++it)
                cout << "Diff: " << it->first << " on " << it->second << endl;
        }
//#endif
            }
    return same;
}

//todo: not sure if is_sorted is set correctly???
template <class IT, class NT>
CSR<IT,NT>::CSR(const string filename): zerobased(true), is_sorted(true), is_value_sorted(false)
{
    IT i;
    bool isUnsy;
    IT num, offset, tmp_nz;
    char *line, *ch;
    FILE *fp;
    IT *col_coo, *row_coo;
    NT *val_coo;
    IT *each_row_index;
    IT *nnz_num;
    const int LINE_LENGTH_MAX = 256;

    isUnsy = false;
    line = (char *)malloc(sizeof(char) * LINE_LENGTH_MAX);
  
    /* Open File */
    fp = fopen(filename.c_str(), "r");
    if(fp == NULL) {
        exit(1);
    }
    do {
        fgets(line, LINE_LENGTH_MAX, fp);
        if (strstr(line, "general")) {
            isUnsy = true;
        }
    } while(line[0] == '%');
  
    /* Get size info */
    sscanf(line, "%d %d %d", &rows, &cols, &tmp_nz);

    /* Store in COO format */
    num = 0;
    col_coo = (IT *)malloc(sizeof(IT) * (tmp_nz));
    row_coo = (IT *)malloc(sizeof(IT) * (tmp_nz));
    val_coo = (NT *)malloc(sizeof(NT) * (tmp_nz));

    while (fgets(line, LINE_LENGTH_MAX, fp)) {
        ch = line;
        /* Read first word (row id)*/
        row_coo[num] = (IT)(atoi(ch) - 1);
        ch = strchr(ch, ' ');
        ch++;
        /* Read second word (column id)*/
        col_coo[num] = (IT)(atoi(ch) - 1);
        ch = strchr(ch, ' ');

        if (ch != NULL) {
            ch++;
            /* Read third word (value data)*/
            val_coo[num] = (NT)atof(ch);
            ch = strchr(ch, ' ');
        }
        else {
            val_coo[num] = 1.0;
        }
        num++;
    }
    fclose(fp);

    /* Count the number of non-zero in each row */
    nnz_num = (IT *)malloc(sizeof(IT) * rows);
    for (i = 0; i < rows; i++) {
        nnz_num[i] = 0;
    }
    for (i = 0; i < num; i++) {
        nnz_num[row_coo[i]]++;
        if(col_coo[i] != row_coo[i] && isUnsy == false) {
            nnz_num[col_coo[i]]++;
            (tmp_nz)++;
        }
    }

    nnz = tmp_nz;

    /* Allocation of rpt, col, val */
    rowptr = (IT *)malloc(sizeof(IT) * (rows + 1));
    colids = (IT *)malloc(sizeof(IT) * (nnz));
    values = (NT *)malloc(sizeof(NT) * (nnz));

    offset = 0;
    for (i = 0; i < rows; i++) {
        rowptr[i] = offset;
        offset += nnz_num[i];
    }
    rowptr[rows] = offset;

    each_row_index = (IT *)malloc(sizeof(IT) * rows);
    for (i = 0; i < rows; i++) {
        each_row_index[i] = 0;
    }
  
    for (i = 0; i < num; i++) {
        colids[rowptr[row_coo[i]] + each_row_index[row_coo[i]]] = col_coo[i];
        values[rowptr[row_coo[i]] + each_row_index[row_coo[i]]++] = val_coo[i];
    
        if (col_coo[i] != row_coo[i] && isUnsy==false) {
            colids[rowptr[col_coo[i]] + each_row_index[col_coo[i]]] = row_coo[i];
            values[rowptr[col_coo[i]] + each_row_index[col_coo[i]]++] = val_coo[i];
        }
    }

    free(line);
    free(nnz_num);
    free(row_coo);
    free(col_coo);
    free(val_coo);
    free(each_row_index);
}

template <class IT, class NT>
void CSR<IT,NT>::print_rows(IT start, IT end)
{
  for (IT i = start; i < end; ++i) {
    cout << i << ":";
    for (IT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      cout << " " << colids[j] << "(" << values[j] << ")";
    }
    cout << endl;
  }
}

template <class IT, class NT>
void CSR<IT, NT>::print_to_file(const string &filename) {
  std::fstream file(filename, std::ios::out);
  for (IT i = 0; i < rows; ++i) {
    file << i << ":";
    for (IT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      file << " " << colids[j] << "(" << values[j] << ")";
    }
    file << endl;
  }
  file.close();
}

template <class IT, class NT>
void CSR<IT, NT>::print_topK(const string &filename, IT topK) {
  assert(is_value_sorted && "CSR must be sorted by values in descending order before picking topK entries");
  std::fstream file(filename, std::ios::out);

  NT eps = 0.0000001f;
  for (IT i = 0; i < rows; ++i) {
//    file << i << ":";
    IT count = 0;
    for (IT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      if(colids[j] == i) continue;
//      if(values[j] < (value_th - eps)) break;

      file << i << " " << colids[j] << " " << values[j] << endl;
      count += 1;
      if(count == topK) break;
    }
//    file << endl;
  }
  file.close();
}

template <class IT, class NT>
void CSR<IT, NT>::print_topK(const string &filename, IT topK, NT value_th) {
  assert(is_value_sorted && "CSR must be sorted by values in descending order before picking topK entries");
  std::fstream file(filename, std::ios::out);
//  IT total_count = 0;
//  for (IT i = 0; i < rows; ++i) {
//    IT degree = (rowptr[i + 1] - rowptr[i]);
//    // skipping diagonal entry
//    if(degree <= topK) {
//      for (IT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
//        if(colids[j] == i) {
//          degree -= 1;
//          break;
//        }
//      }
//    }
//    total_count += min(topK, degree);
//  }
//  file << total_count << endl;
  NT eps = 0.0000001f;
  for (IT i = 0; i < rows; ++i) {
//    file << i << ":";
    IT count = 0;
    for (IT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      if(colids[j] == i) continue;
      if(values[j] < (value_th - eps)) break;

      file << i << " " << colids[j] << " " << values[j] << endl;
      count += 1;
      if(count == topK) break;
    }
//    file << endl;
  }
  file.close();
}

template <class IT, class NT>
void CSR<IT,NT>::shuffleIds()
{
    mt19937_64 mt(0);
    for (IT i = 0; i < rows; ++i) {
        IT offset = rowptr[i];
        IT width = rowptr[i + 1] - rowptr[i];
        uniform_int_distribution<IT> rand_scale(0, width - 1);
        for (IT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            IT target = rand_scale(mt);
            IT tmpId = colids[offset + target];
            NT tmpVal = values[offset + target];
            colids[offset + target] = colids[j];
            values[offset + target] = values[j];
            colids[j] = tmpId;
            values[j] = tmpVal;
        }
    }
    is_sorted = false;
    is_value_sorted = false;
}

template <class IT, class NT>
NT CSR<IT,NT>::overlap_coefficient_similarity(int a, int b) {
  int c = 0;
  set<IT> sb;
  for (IT j = rowptr[a]; j < rowptr[a+1]; ++j) sb.insert(colids[j]);
  for (IT j = rowptr[b]; j < rowptr[b+1]; ++j) {
    if (sb.find(colids[j]) != sb.end()) c++;
  }
  int u = min((rowptr[a+1] - rowptr[a]), (rowptr[b+1] - rowptr[b]));

  return (u == 0) ? 0.0 : 1.0 * c / u;
}

template <class IT, class NT>
NT CSR<IT,NT>::calculate_average_oc_similarity()
{
  NT th = 0.0;
//#pragma omp parallel for
  for (IT i = 0; i < rows-1; ++i)
  {
    th += overlap_coefficient_similarity(i, i+1);
  }
  if(rows > 1) th /= (rows - 1);
  return th;
}

template <class IT, class NT>
NT CSR<IT,NT>::jaccard_similarity(int a, int b) {
  int c = 0;
  set<IT> sb;
  for (IT j = rowptr[a]; j < rowptr[a+1]; ++j) sb.insert(colids[j]);
  for (IT j = rowptr[b]; j < rowptr[b+1]; ++j) {
    if (sb.find(colids[j]) != sb.end()) c++;
  }
  int u = (rowptr[a+1] - rowptr[a]) + (rowptr[b+1] - rowptr[b]) - c;

  return (u == 0) ? 0.0 : 1.0 * c / u;
}

template <class IT, class NT>
NT CSR<IT,NT>::clustering_factor(int a, int b) {
  int c = 0;
  set<IT> sb;
  for (IT j = rowptr[a]; j < rowptr[a+1]; ++j) sb.insert(colids[j]);
  for (IT j = rowptr[b]; j < rowptr[b+1]; ++j) {
    if (sb.find(colids[j]) != sb.end()) c++;
  }
  int u = (rowptr[a+1] - rowptr[a]) + (rowptr[b+1] - rowptr[b]) - c;

  NT jacc = (u == 0) ? 0.0 : 1.0 * c / u;
  NT annz = ((rowptr[a+1] - rowptr[a]) + (rowptr[b+1] - rowptr[b])) / 2.0;
  return (u / annz / jacc);
}

template <class IT, class NT>
IT CSR<IT,NT>::common_elements(int a, int b) {
    int c = 0;
    set<IT> sb;
    for (IT j = rowptr[a]; j < rowptr[a+1]; ++j) sb.insert(colids[j]);
    for (IT j = rowptr[b]; j < rowptr[b+1]; ++j) {
        if (sb.find(colids[j]) != sb.end()) c++;
    }
//    int u = (rowptr[a+1] - rowptr[a]) + (rowptr[b+1] - rowptr[b]) - c;

//    return (u == 0) ? 0.0 : 1.0 * c / u;
    return c;
}

template <class IT, class NT>
NT CSR<IT,NT>::calculate_average_similarity()
{
  NT th = 0.0;
//#pragma omp parallel for
  for (IT i = 0; i < rows-1; ++i)
  {
    th += jaccard_similarity(i, i+1);
  }
  if(rows > 1) th /= (rows - 1);
  return th;
}

template <class IT, class NT>
vector<NT> CSR<IT,NT>::calculate_average_similarity(IT ngram)
{
  vector<NT> th(ngram, 0.0);
//#pragma omp parallel for
  for (IT i = 0; i < rows; ++i)
  {
    for(IT j=1; j<=ngram; j+=1) {
      if(i+j < rows) th[j-1] += jaccard_similarity(i, i+j);
    }
  }
  for(IT j=1; j<=ngram; j+=1) {
//    cout << th[j - 1] << " ";
    if(rows > j) th[j - 1] /= (rows - j);
//    cout << th[j - 1] << endl;
  }
  return th;
}

template <class IT, class NT>
std::tuple<NT, NT, NT> CSR<IT, NT>::calculate_similarity_stats() {
  std::vector<NT> similarities;
  similarities.reserve(rows - 1);

  NT sum = 0.0;
  NT log_sum = 0.0;

  for (IT i = 0; i < rows - 1; ++i) {
    NT sim = jaccard_similarity(i, i + 1);
    similarities.push_back(sim);
    sum += sim;
    if (sim > 0) log_sum += std::log(sim); // avoid log(0)
  }

  NT average = (rows > 1) ? (sum / (rows - 1)) : 0.0;
  NT geo_mean = (rows > 1) ? std::exp(log_sum / (rows - 1)) : 0.0;

  std::sort(similarities.begin(), similarities.end());
  NT median;
  size_t n = similarities.size();
  if (n == 0) {
    median = 0.0;
  } else if (n % 2 == 0) {
    median = (similarities[n / 2 - 1] + similarities[n / 2]) / 2.0;
  } else {
    median = similarities[n / 2];
  }

  return std::make_tuple(average, geo_mean, median);
}

template <class IT, class NT>
void CSR<IT,NT>::gather_unique_col_ids_in_cluster(unordered_set<IT> &unique_cols, IT st, IT nd) {
  for(IT i=st; i<nd; i+=1) {
    for (IT j = rowptr[i]; j < rowptr[i+1]; ++j) unique_cols.insert(colids[j]);
  }
}

template <class IT, class NT>
NT CSR<IT,NT>::calculate_average_unique_cols_in_cluster(IT cluster_sz)
{
  NT count = 0.0;
  unordered_set<IT> unique_cols;
//#pragma omp parallel for
  for (IT i = 0; i < rows; i+=cluster_sz)
  {
    gather_unique_col_ids_in_cluster(unique_cols, i, min(i + cluster_sz, rows));
    count += unique_cols.size();
    unique_cols.clear();
  }
  IT num_clusters = (rows / cluster_sz) + ((rows % cluster_sz != 0) ? 1 : 0);
  return count / num_clusters;
}

template <class IT, class NT>
NT CSR<IT,NT>::jaccard_similarity(const unordered_set<IT> &a, const unordered_set<IT> &b) {
  int c = 0;
  for (IT lookup : a) {
    if (b.find(lookup) != b.end()) c++;
  }
  int u = a.size() + b.size() - c;

  return (u == 0) ? 0.0 : 1.0 * c / u;
}

template <class IT, class NT>
NT CSR<IT,NT>::calculate_average_similarity_in_cluster(IT cluster_sz)
{
  assert(cluster_sz < rows && "Cluster size can't be larger than number of available rows");
  NT th = 0.0;
  IT num_clusters = 0;
  unordered_set<IT> prev, curr;

  gather_unique_col_ids_in_cluster(prev, 0, cluster_sz);
//#pragma omp parallel for
  for (IT i = cluster_sz; i < rows; i+=cluster_sz)
  {
    gather_unique_col_ids_in_cluster(curr, i, min(i+cluster_sz, rows));
    th += jaccard_similarity(prev, curr);

    prev.clear();
    prev = curr;
    curr.clear();
    num_clusters += 1;
  }
  th /= num_clusters;
  return th;
}

template <class IT, class NT>
pair<IT, NT> CSR<IT,NT>::calculate_bandwidth() {
  IT bw = 0;
  NT abw = 0.0;
  IT valid_rows = 0;
  IT lft, rgt;
  IT degree;
  assert(is_sorted && "CSR needs to be sorted to calculate the bandwidth. Abort!");
  for(IT i=0; i<rows; i+=1) {
    degree = rowptr[i+1] - rowptr[i];
    if(degree > 0) {
      lft = abs(colids[rowptr[i]] - i);
      rgt = abs(colids[rowptr[i+1] - 1] - i);

      bw = max(max(bw, lft), rgt);
      abw += (lft + rgt);
      valid_rows += 1;
    }
  }
//  return pair<IT, NT>(bw, abw/valid_rows);
  return pair<IT, NT>(bw, (NT) bw/cols);
}

template <class IT, class NT>
pair<IT, NT> CSR<IT,NT>::calculate_profile() {
  IT prof = 0;
  IT valid_rows = 0;
  IT degree;
  assert(is_sorted && "CSR needs to be sorted to calculate the profile. Abort!");
  for(IT i=0; i<rows; i+=1) {
    degree = rowptr[i+1] - rowptr[i];
    if(degree > 0 && colids[rowptr[i]] <= i) {
      prof += abs(colids[rowptr[i]] - i);
      valid_rows += 1;
    }
  }
//  return pair<IT, NT>(prof, (NT)prof / valid_rows / cols / 2);
  return pair<IT, NT>(prof, (NT)prof / nnz / 2);
}

//template <class IT, class NT>
//pair<IT, NT> CSR<IT,NT>::calculate_profile_V1() {
//  IT prof = 0;
//  IT valid_rows = 0;
//  IT lft, rgt;
//  IT degree;
//  assert(is_sorted && "CSR needs to be sorted to calculate the profile. Abort!");
//  for(IT i=0; i<rows; i+=1) {
//    degree = rowptr[i+1] - rowptr[i];
//    if(degree > 0) {
//      lft = abs(colids[rowptr[i]] - i);
//      rgt = abs(colids[rowptr[i+1] - 1] - i);
//
//      if(colids[rowptr[i]] <= i) prof += abs(colids[rowptr[i]] - i);
//      if(colids[rowptr[i + i] - 1] > i) prof += abs(colids[rowptr[i]] - i);
//      valid_rows += 1;
//    }
//  }
////  return pair<IT, NT>(prof, (NT)prof / valid_rows / cols / 2);
//  return pair<IT, NT>(prof, (NT)prof / nnz / 2);
//}

template <class IT, class NT>
NT CSR<IT,NT>::calculate_average_row_width() {
  IT degree;
  IT valid_rows = 0;
  NT width = 0.0;
  assert(is_sorted && "CSR needs to be sorted to calculate the average row width. Abort!");
  for(IT i=0; i<rows; i+=1) {
    degree = rowptr[i+1] - rowptr[i];
    if(degree > 1) {
      width += (colids[rowptr[i+1] - 1] - colids[rowptr[i]]);
      valid_rows += 1;
    }
  }
  width /= valid_rows;
  return width;
}

template <class IT, class NT>
pair<IT, NT> CSR<IT,NT>::calculate_offdiagonal_nonzero_count(int block_sz) {
  assert(rows == cols && "CSR needs to be square to calculate the off-diagonal non-zero counts. Abort!");
  IT count = 0;
  IT block_st = 0, block_nd = block_sz;
  IT num_blocks = (rows / block_sz) + ((rows % block_sz) == 0 ? 0 : 1);

  for(IT b=0; b<num_blocks; b+=1) {
    for(IT i=block_st; i<block_nd; i+=1) {
      for (IT j = rowptr[i]; j < rowptr[i+1]; ++j) {
        if(colids[j] < block_st || colids[j] >= block_nd) count += 1;
      }
    }
    block_st += block_sz;
    block_nd += block_sz;
    if(block_nd > rows) block_nd = rows;
  }

  return pair<IT, NT>(count, (NT)count / nnz);
}

/* Get total number of floating operations and average
 * then, use it for assigning rows to thread as the amount of work is equally distributed
 */
template <class IT, class NT>
NT CSR<IT, NT>::get_load_imbalance_factor(const IT thread_num, const CSR<IT,NT> & B)
{
  IT total_intprod = 0;
  IT *row_nz = my_malloc<IT>(rows);
#pragma omp parallel
  {
    IT each_int_prod = 0;
#pragma omp for
    for (IT i = 0; i < rows; ++i) {
      IT nz_per_row = 0;
      for (IT j = rowptr[i]; j < rowptr[i + 1]; ++j) {    // col-ids of A
        // todo: do we need to consider B matrix here?
        nz_per_row += B.rowptr[colids[j] + 1] - B.rowptr[colids[j]];    // summing up rows of B
      }
      row_nz[i] = nz_per_row;
      each_int_prod += nz_per_row;
    }
#pragma omp atomic
    total_intprod += each_int_prod;
  }

  IT *ps_row_nz = my_malloc<IT>(rows + 1);
  /* Prefix sum of #intermediate products */
  scan(row_nz, ps_row_nz, rows + 1);

  IT current_intprod, average_intprod = (total_intprod + thread_num - 1) / thread_num;
  IT intprod_gap;
  // long long int average_intprod = total_intprod / thread_num;
  NT load_imbalance = 0.0;

  /* Search end point of each range */
  auto last_itr = 0;
  for(int tid=0; tid<thread_num-1; tid+=1)
  {
//    int tid = omp_get_thread_num();
    auto curr_end_itr = (lower_bound(ps_row_nz, ps_row_nz + rows + 1, average_intprod * (tid + 1))) - ps_row_nz;
    current_intprod = ps_row_nz[curr_end_itr] - ps_row_nz[last_itr];
    intprod_gap = abs(current_intprod - average_intprod);
    // todo: what happen if current_intprod < average_intprod???
    load_imbalance = max(load_imbalance, ((NT) intprod_gap / average_intprod) * 100.0);
    last_itr = curr_end_itr;
//    rows_offset[tid + 1] = end_itr;
    // if (tid == thread_num - 1) rows_offset[tid + 1] = rows;
  }
//  rows_offset[thread_num] = rows;
  my_free<IT>(row_nz);
  my_free<IT>(ps_row_nz);

  return load_imbalance;
}

template <class IT, class NT>
void CSR<IT,NT>::print_jaccard_similarity(const string& filename)
{
  NT js, MAX_JS = 0.0;
  IT MAX_NZ = 0;
  IT count = 0;
//  std::fstream file(filename, std::ios::out);
  for (IT i = 0; i < rows-1; ++i)
  {
    js = jaccard_similarity(i, i+1);
    MAX_JS = max(js, MAX_JS);
    MAX_NZ = max(MAX_NZ, max((rowptr[i+1] - rowptr[i]), (rowptr[i+1+1] - rowptr[i+1])));
    if(js >= 0.5) count += 1;
//    if(js >= 0.5 && ((rowptr[i+1] - rowptr[i]) > 100 || (rowptr[i+1+1] - rowptr[i+1]) > 100)) file << js << "\t" << (rowptr[i+1] - rowptr[i]) << "\t" << (rowptr[i+1+1] - rowptr[i+1]) << "\t" << cols << endl;
//    if(js >= 0.5) file << js << "\t" << (rowptr[i+1] - rowptr[i]) << "\t" << (rowptr[i+1+1] - rowptr[i+1]) << "\t" << cols << endl;
//    file << js << " " << (rowptr[i+1] - rowptr[i]) << " " << (rowptr[i+1+1] - rowptr[i+1]) << " " << cols << endl;
  }
//  file.close();
  cout << "\nCSR::print_jaccard_similarity()" << endl;
  cout << "MAX-NZ: " << MAX_NZ << endl;
  cout << "MAX-jaccard-similarity-score: " << MAX_JS << endl;
  cout << "Count of consecutive rows contains more than 0.5 score: " << count << endl;
}

/// Copied code from MaskedSpGEMM
template <class IT, class NT>
void CSR<IT,NT>::sortIds()
{
#pragma omp parallel for
  for (IT i = 0; i < rows; ++i)
  {
    vector< pair<IT,NT> > tosort;
    for (IT j = rowptr[i]; j < rowptr[i+1]; ++j)
    {
      tosort.push_back(make_pair(colids[j], values[j]));
    }
    std::sort(tosort.begin(), tosort.end());
    auto begitr = tosort.begin();
    for (IT j = rowptr[i]; j < rowptr[i+1]; ++j)
    {
      colids[j] = begitr->first;
      values[j] = begitr->second;
      ++begitr;
    }
  }
  is_sorted = true;
  is_value_sorted = false;
}

template <class IT, class NT>
void CSR<IT,NT>::sortValues()
{
#pragma omp parallel for
  for (IT i = 0; i < rows; ++i)
  {
    vector< pair<NT, IT> > tosort;
    for (IT j = rowptr[i]; j < rowptr[i+1]; ++j)
    {
      tosort.push_back(make_pair(values[j], colids[j]));
    }
    std::sort(tosort.begin(), tosort.end(), sort_large<IT, NT>);
    auto begitr = tosort.begin();
    for (IT j = rowptr[i]; j < rowptr[i+1]; ++j)
    {
      colids[j] = begitr->second;
      values[j] = begitr->first;
      ++begitr;
    }
  }
  is_sorted = false;
  is_value_sorted = true;
}

/// Copied code from spmm
template <class IT, class NT>
map<IT, map<IT, NT>> CSR<IT,NT>::get_rowmap() {
//  vector<int> rowptr_cpu(rows+1);
//  vector<int> colidx_cpu(nnz);
//  vector<T> values_cpu(nnz);
//  cudaMemcpy(rowptr_cpu.data(), rowptr, (rows+1)*sizeof(int), cudaMemcpyDeviceToHost);
//  cudaMemcpy(colidx_cpu.data(), colidx, (nnz)*sizeof(int), cudaMemcpyDeviceToHost);
//  cudaMemcpy(values_cpu.data(), values, (nnz)*sizeof(T), cudaMemcpyDeviceToHost);


  map<IT, map<IT, NT>> rowlist;
  for (int i=0; i<rows; i++) {
    for (int j=rowptr[i]; j<rowptr[i+1]; j++) {
      if (rowlist.find(i) == rowlist.end()) rowlist[i] = map<IT, NT>();
      rowlist[i][colids[j]] = values[j];
    }
  }
  return rowlist;
}

template <class IT, class NT>
void CSR<IT, NT>::loadSplittedMatrix(vector<pair<IT, map<IT, NT>>> &csr, vector<IT> &reordered_rows, int panel_size) {
  assert (csr.size() == reordered_rows.size());
  rows = csr.size();

  vector<int> rowptr_cpu;
  vector<int> rowidx_cpu;
  vector<int> colidx_cpu;
  vector<NT> values_cpu;
  rowptr_cpu.push_back(0);
  nnz = 0;


  for (int i=0; i<rows; i+=panel_size) {
    map<int, vector<int>> collist;
    for (int j=i; j<(i+panel_size>rows?rows:i+panel_size); j++) {
      rowidx_cpu.push_back(csr[reordered_rows[j]].first);
      for (auto &nz: csr[reordered_rows[j]].second) {
        int c = nz.first;
        if (collist.find(c) == collist.end()) collist[c] = vector<int>();
        collist[c].push_back(j-i);
      }
    }
    vector<pair<int, vector<int>>> collist_vec;
    copy(collist.begin(), collist.end(), back_inserter<vector<pair<int, vector<int>>>>(collist_vec));
    sort(collist_vec.begin(), collist_vec.end(), [](pair<int, vector<int>> &a, pair<int, vector<int>> &b) {return a.second.size() > b.second.size(); } );
    vector<vector<int>> colidx_tmp(panel_size);
    for (int j=0; j<collist_vec.size(); j++) {
      for (int rr: collist_vec[j].second) {
        colidx_tmp[rr].push_back(collist_vec[j].first);
      }
    }
    for (int j=0; i+j<(i+panel_size>rows?rows:i+panel_size); j++) {
      for (int cc: colidx_tmp[j]) {
        colidx_cpu.push_back(cc);
        NT v = (csr[reordered_rows[j+i]].second)[cc];
        //cout << v << endl;
        values_cpu.push_back(v);
      }
      nnz += colidx_tmp[j].size();
      rowptr_cpu.push_back(nnz);
    }
  }

  assert (nnz == colidx_cpu.size());

  if (nnz > 0) {
    rowptr = my_malloc<IT>(rows + 1);
    colids = my_malloc<IT>(nnz);
    values = my_malloc<NT>(nnz);
//    cudaMalloc(&rowptr, sizeof(int)*(nrows+1));
//    cudaMalloc(&rowidx, sizeof(int)*(nrows));
//    cudaMalloc(&colidx, sizeof(int)*(nnz));
//    cudaMalloc(&values, sizeof(int)*(nnz));
//    cudaCheckError();

    copy(rowptr_cpu.begin(), rowptr_cpu.end(), rowptr);
    copy(colidx_cpu.begin(), colidx_cpu.end(), colids);
    copy(values_cpu.begin(), values_cpu.end(), values);
//    cudaMemcpy(rowptr, rowptr_cpu.data(), (nrows+1)*sizeof(int), cudaMemcpyHostToDevice);
//    cudaCheckError();
//    cudaMemcpy(colidx, colidx_cpu.data(), (nnz)*sizeof(int), cudaMemcpyHostToDevice);
//    cudaCheckError();
//    cudaMemcpy(values, values_cpu.data(), (nnz)*sizeof(T), cudaMemcpyHostToDevice);
//    cudaCheckError();

    // todo: I am not sure whether we will need the rowidx or not
//    cudaMemcpy(rowidx, rowidx_cpu.data(), (nrows)*sizeof(int), cudaMemcpyHostToDevice);
//    cudaCheckError();
  }
}

template <class IT, class NT>
void CSR<IT,NT>::load_reorderedCSR(map<IT, map<IT, NT>> &csr, vector<IT> &reordered_rows, int panel_size, map<IT, map<IT, NT>> &spp3) {

  assert (csr.size() == reordered_rows.size());
  rows = csr.size();

  vector<int> rowptr_cpu;
  vector<int> rowidx_cpu;
  vector<int> colidx_cpu;
  vector<NT> values_cpu;
  rowptr_cpu.push_back(0);
  nnz = 0;


  for (int i=0; i<rows; i+=panel_size) {
    map<int, vector<int>> collist;    // raqib: map col-id to vector<row-ids> where col-id belong
    for (int j=i; j<(i+panel_size>rows?rows:i+panel_size); j++) {
      for (auto &nz: csr[reordered_rows[j]]) {
        int c = nz.first;
        if (collist.find(c) == collist.end()) collist[c] = vector<int>();
        collist[c].push_back(j-i);
      }
    }
    vector<pair<int, vector<int>>> collist_vec;   // raqib: vector version of @collist
    copy(collist.begin(), collist.end(), back_inserter<vector<pair<int, vector<int>>>>(collist_vec));
    // raqib: sorting col-ids based on the vector<row-ids>.size()
    sort(collist_vec.begin(), collist_vec.end(), [](pair<int, vector<int>> &a, pair<int, vector<int>> &b) {return a.second.size() > b.second.size(); } );
    vector<vector<int>> colidx_tmp(panel_size);   // raqib: @colidx_tmp is the inverse mapping of @collist_vec (row-id to col-ids)
    for (int j=0; j<collist_vec.size(); j++) {
      for (int rr: collist_vec[j].second) {
        colidx_tmp[rr].push_back(collist_vec[j].first);
      }
    }
    for (int j=0; i+j<(i+panel_size>rows?rows:i+panel_size); j++) {
      // splitting rows which have (nz > 3000) and moving those entries from @csr to @spp3
      while (colidx_tmp[j].size() > 3000) {
        int r = reordered_rows[i+j];
        int c = colidx_tmp[j].back();
        NT v = csr[r][c];
        if (spp3.find(r) == spp3.end()) {
          spp3[r] = map<int, NT>();
        }
        spp3[r][c] = v;
        colidx_tmp[j].pop_back();
      }
      for (int cc: colidx_tmp[j]) {
        colidx_cpu.push_back(cc);
        NT v = csr[reordered_rows[j+i]][cc];
        //cout << v << endl;
        values_cpu.push_back(v);
      }
      nnz += colidx_tmp[j].size();
      rowptr_cpu.push_back(nnz);
    }
  }

  assert (nnz == colidx_cpu.size());

  if (nnz > 0) {
    rowptr = my_malloc<IT>(rows + 1);
    colids = my_malloc<IT>(nnz);
    values = my_malloc<NT>(nnz);
//    cudaMalloc(&rowptr, sizeof(int)*(rows+1));
//    cudaMalloc(&rowidx, sizeof(int)*(rows));
//    cudaMalloc(&colidx, sizeof(int)*(nnz));
//    cudaMalloc(&values, sizeof(int)*(nnz));
//    cudaCheckError();

    copy(rowptr_cpu.begin(), rowptr_cpu.end(), rowptr);
    copy(colidx_cpu.begin(), colidx_cpu.end(), colids);
    copy(values_cpu.begin(), values_cpu.end(), values);
//    cudaMemcpy(rowptr, rowptr_cpu.data(), (rows+1)*sizeof(int), cudaMemcpyHostToDevice);
//    cudaCheckError();
//    cudaMemcpy(colidx, colidx_cpu.data(), (nnz)*sizeof(int), cudaMemcpyHostToDevice);
//    cudaCheckError();
//    cudaMemcpy(values, values_cpu.data(), (nnz)*sizeof(T), cudaMemcpyHostToDevice);
//    cudaCheckError();

    // todo: I am not sure whether we will need the rowidx or not
//    cudaMemcpy(rowidx, reordered_rows.data(), (rows)*sizeof(int), cudaMemcpyHostToDevice);
//    cudaCheckError();
  }
}

#endif
