#ifndef B2FRENDER_B2F_FEATURES_HPP
#define B2FRENDER_B2F_FEATURES_HPP


#include <memory>

struct MKSHARED
{
    template<class T>
    std::shared_ptr<T> operator*(T* ptr)
    {
        return std::shared_ptr<T>(ptr);
    }
} MK_SHARED;
#define shared MK_SHARED*


#define CTORS /*CTORS*/
#define CONSTANTS /*CONSTANTS*/
#define METHODS /*METHODS*/
#define OPERATORS /*OPERATORS*/
#define FIELDS /*FIELDS*/

#endif //B2FRENDER_B2F_FEATURES_HPP
