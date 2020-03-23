
#pragma once

#include <map>
#include <iostream>

namespace sphericalsfm {
    template<typename T>
    class SparseVector : public std::map<int, T>
    {
    public:
        typedef typename std::map<int,T>::iterator Iterator;
        typedef typename std::map<int,T>::const_iterator ConstIterator;
        
        T& operator()( int r ) { return (*this)[r]; }
        const T& operator()( int r ) const { return (*this)[r]; }

        bool exists( int r ) const
        {
            return ( this->find(r) != this->end() );
        }
    };

    template<typename T>
    class SparseMatrix : public SparseVector< SparseVector<T> >
    {
    public:
        typedef typename SparseVector< SparseVector<T> >::Iterator RowIterator;
        typedef typename SparseVector< SparseVector<T> >::ConstIterator ConstRowIterator;

        typedef typename SparseVector<T>::Iterator ColumnIterator;
        typedef typename SparseVector<T>::ConstIterator ConstColumnIterator;

        T& operator()( int r, int c ) { return (*this)[r][c]; }
        const T& operator()( int r, int c ) const { return (*this)[r][c]; }
        
        bool exists( int r, int c ) const
        {
            ConstRowIterator rowit = this->find(r);
            if ( rowit == this->end() ) return false;
            ConstColumnIterator colit = rowit->second.find(c);
            return ( colit != rowit->second.end() );
        }
        
        // not sure why this is necessary
        void erase( int r )
        {
            SparseVector< SparseVector<T> >::erase( r );
        }
        
        void erase( int r, int c )
        {
            RowIterator rowit = this->find(r);
            if ( rowit == this->end() ) return;
            ColumnIterator colit = rowit->second.find(c);
            if ( colit == rowit->second.end() ) return;
            rowit->second.erase(colit);
        }
    };
}
