#ifndef OUTSTREAMBUF_H
#define OUTSTREAMBUF_H

#include <fstream>
#include <streambuf>
#include <iostream>

/* write simultaneously to cout/cerr and a file */
class outstreambuf : public std::streambuf {
public:
    outstreambuf* open( const char* name, std::ios::openmode mode = std::ios::out);
    bool is_open();
    outstreambuf* close();
    ~outstreambuf() { close(); }
    std::streambuf *get_fout_buf() {
        return fout_buf;
    }
    std::streambuf *get_cout_buf() {
        return cout_buf;
    }
    std::ofstream *get_fout() {
        return &fout;
    }
    
protected:
    std::ofstream fout;
    std::streambuf *cout_buf;
    std::streambuf *fout_buf;
    virtual int     overflow( int c = EOF);
    virtual int     sync();
};

#endif // OUTSTREAMBUF_H