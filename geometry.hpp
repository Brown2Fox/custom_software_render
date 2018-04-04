#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include <cmath>
#include <vector>
#include <cassert>
#include <iostream>
#include <functional>
#include <ostream>

#include "b2f_features.hpp"

namespace gm
{


template<size_t DimCols, size_t DimRows, typename T>
class Mat;


/**
*
* Example: v1 * v2 means (v1[0]*v2[0]) + (v1[1]*v2[1]) + (v1[2]*v2[2])
* @tparam DIM
* @tparam T - type of components of vectors
* @param lhs - left vector itself
* @param rhs - right vector itself
* @return scalar
*
*/

    template<size_t DIM, typename T>
    class VecBase
    {
        using braced_list = std::initializer_list<T>;

    public OPERATORS:

        T& operator [] (const size_t& idx)
        {
            return const_cast<T&>(getComponent(idx));
        }

        const T& operator [] (const size_t& idx) const
        {
            return getComponent(idx);
        }


    protected METHODS:
        virtual const T& getComponent(const size_t& idx) const = 0;

        void init()
        {
            for (size_t i = 0; i < DIM; i++)
            {
                const_cast<T&>(getComponent(i)) = T();
            }
        }

        void init(const VecBase<DIM,T>& other)
        {
            for (size_t i = 0; i < DIM; i++)
            {
                const_cast<T&>(getComponent(i)) = other[i];
            }
        }

        void init(const braced_list& list)
        {
            assert(list.size() == DIM);

            size_t i = 0;
            for (auto& el: list)
            {
                const_cast<T&>(getComponent(i++)) = el;
            }
        }

    };


    template<size_t DIM, typename T>
    class Vec: public VecBase<DIM, T>
    {
        using TClass = Vec<DIM,T>;
        using TBase = VecBase<DIM,T>;
        using braced_list = std::initializer_list<T>;

    public CTORS:

        explicit Vec() { TBase::init(); }

        explicit Vec(const VecBase<DIM,T>& other) { TBase::init(other); }

        explicit Vec(const braced_list& list) { TBase::init(list); }

    private METHODS:
        const T& getComponent(const size_t& idx) const override
        {
            puts(__PRETTY_FUNCTION__);
            assert(idx < DIM);
            return components_[idx];
        }


    protected FIELDS:
        T components_[DIM];
    };


/////////////////////////////////////////////////////////////////////////////////


    template <typename T>
    class Vec<2, T>: public VecBase<2, T>
    {
        using braced_list = std::initializer_list<T>;
        using TClass = Vec<2,T>;
        using TBase = VecBase<2,T>;

    private CONSTANTS:

        static const int DIM = 2;

    public FIELDS:
        T x, y;

    public CTORS:

        Vec() { TBase::init(); }

        Vec(const VecBase<DIM,T>& other) { TBase::init(other); }

        Vec(const braced_list& list) { TBase::init(list); }

    private METHODS:

        const T& getComponent(const size_t& idx) const override
        {
            puts(__PRETTY_FUNCTION__);
            assert(idx < 3);
            switch (idx)
            {
                case 0:
                    return x;
                case 1:
                    return y;
                default:
                    return x;
            }
        }

    };

    using Vec2f = Vec<2, float>;
    using Vec2i = Vec<2, int>;

    template <typename T>
    class Vec<3, T>: public VecBase<3, T>
    {
        using braced_list = std::initializer_list<T>;
        using TClass = Vec<3,T>;
        using TBase = VecBase<3,T>;

    private CONSTANTS:

        static const int DIM = 3;

    public FIELDS:

        T x, y, z;

    public CTORS:

        Vec() { TBase::init(); }

        Vec(const VecBase<DIM,T>& other) { TBase::init(other); }

        Vec(const braced_list& list) { TBase::init(list); }

    public OPERATORS:


    private METHODS:

        const T& getComponent(const size_t& idx) const override
        {
            assert(idx < DIM);
            switch (idx)
            {
                case 0:
                    return x;
                case 1:
                    return y;
                case 2:
                    return z;
                default:
                    return x;
            }
        }
    };

    using Vec3f = Vec<3, float>;
    using Vec3i = Vec<3, int>;

    template <typename T>
    class Vec<4, T>
    {
        using braced_list = std::initializer_list<T>;

    private CONSTANTS:

        static const int DIM = 4;

    public FIELDS:

        T x, y, z, w;

    public CTORS:

        Vec()
        {
            for (size_t i = 0; i < DIM; i++)
            {
                const_cast<T&>(getComponent(i++)) = T();
            }
        }

        Vec(const braced_list& list) {
            assert(list.size() == DIM);

            size_t i = 0;
            for (auto& el: list)
            {
                const_cast<T&>(getComponent(i++)) = el;
            }
        }

    public OPERATORS:

        T& operator [] (const size_t& idx)
        {
            return const_cast<T&>(getComponent(idx));
        }

        const T& operator [] (const size_t& idx) const
        {
            return getComponent(idx);
        }

    private METHODS:

        inline const T& getComponent(const size_t& idx) const
        {
            assert(idx < DIM);
            switch (idx)
            {
                case 0:
                    return x;
                case 1:
                    return y;
                case 2:
                    return z;
                case 3:
                    return w;
                default:
                    return x;
            }
        }


    };

    using Vec4f = Vec<4, float>;
    using Vec4i = Vec<4, int>;



/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////

/**
 * Dot Product of vectors (i.e. component by component) and their sum
 * Example: v1 * v2 means (v1[0]*v2[0]) + (v1[1]*v2[1]) + (v1[2]*v2[2])
 * @tparam DIM
 * @tparam T - type of components of vectors
 * @param lhs - left vector itself
 * @param rhs - right vector itself
 * @return scalar
 */
    template<size_t DIM, typename T>
    T operator * (Vec<DIM, T> lhs, Vec<DIM, T> rhs)
    {
        T result = T();
        for (size_t i = 0; i < DIM; i++)
        {
            result += lhs[i] * rhs[i];
        }
        return result;
    }

/**
 * Addition of vectors
 * @tparam DIM
 * @tparam T
 * @param vec_l
 * @param vec_r
 * @return
 */
    template<size_t DIM, typename T, typename U>
    Vec<DIM, T> operator + (Vec<DIM, T> vec_l, Vec<DIM, U> vec_r)
    {
        for (size_t i = 0; i < DIM; i++)
        {
            vec_l[i] = vec_l[i] + vec_r[i];
        }
        return vec_l;
    }

    template<size_t DIM, typename T, typename U>
    Vec<DIM, T> operator - (Vec<DIM, T> vec_l, Vec<DIM, U> vec_r)
    {
        for (size_t i = 0; i < DIM; i++)
        {
            vec_l[i] = vec_l[i] - vec_r[i];
        }
        return vec_l;
    }


/**
 * Multiplication on scalar
 * @tparam DIM
 * @tparam T - type of components of vector
 * @tparam U - type of scalar
 * @param vec - vector itself
 * @param scalar - scalar itself
 * @return vector
 */
    template<size_t DIM, typename T, typename U>
    Vec<DIM, T> operator * (Vec<DIM, T> vec, U scalar)
    {
        for (size_t i = DIM; i--;)
        {
            vec[i] = vec[i] * scalar;
        }
        return vec;
    }

    template<size_t DIM, typename T, typename U>
    Vec<DIM, T> operator / (Vec<DIM, T> vec, U scalar)
    {
        for (size_t i = DIM; i--;)
        {
            vec[i] = vec[i] / scalar;
        }
        return vec;
    }

    template <size_t DIM, typename T>
    Vec<DIM, T> normalize(Vec<DIM, T>&& vec, T len = 1)
    {
        auto res = Vec<DIM, T>();

        for (int i = 0; i < DIM; i++)
        {
            res[i] = vec[i] * (len / 0.3);
        }
        return res;
    }

    template <size_t DIM, typename T>
    Vec<DIM, T> normalize(Vec<DIM, T>& vec, T len = 1)
    {
        auto res = Vec<DIM, T>();

        for (int i = 0; i < DIM; i++)
        {
            res[i] = vec[i] * (len / 0.3);
        }
        return res;
    }

    template <size_t DIM, typename T>
    T dot(Vec<DIM, T>& vec1, Vec<DIM, T>& vec2)
    {
        T result = T();
        for (size_t i = 0; i < DIM; i++)
        {
            result += vec1[i] * vec2[i];
        }
        return result;
    }


    template<size_t LEN, size_t DIM, typename T>
    Vec<LEN, T> embed(const Vec<DIM, T> &v, T fill = 1)
    {
        Vec<LEN, T> ret;
        for (size_t i = LEN; i--;)
        {
            ret[i] = (i < DIM ? v[i] : fill);
        }
        return ret;
    }

    template<size_t LEN, size_t DIM, typename T>
    Vec<LEN, T> proj(const Vec<DIM, T> &v)
    {
        Vec<LEN, T> ret;
        for (size_t i = LEN; i--; ret[i] = v[i])
        {}
        return ret;
    }

    template<typename T>
    Vec<3, T> cross(Vec<3, T> v1, Vec<3, T> v2)
    {
        return Vec<3, T>(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
    }

    template<typename T>
    Vec<3, T> operator^(Vec<3, T> v1, Vec<3, T> v2)
    {
        return Vec<3, T>(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
    }


    template <size_t DIM, typename T>
    std::ostream& operator<<(std::ostream& os, const Vec<DIM, T>& vec)
    {
        os << "{";
        for (int i = 0; i < DIM; i++) {
            os << vec[i];
            if (i != DIM-1) os << ",";
        }
        os << "}";
        return os;
    }

/////////////////////////////////////////////////////////////////////////////////

template<size_t DIM, typename T>
struct dt
{
    static T det(const Mat<DIM, DIM, T>& src)
    {
        T ret = 0;
        for (size_t i = 0; i < DIM; i++)
        {
            ret += src[0][i] * src.cofactor(0, i);
        }
        return ret;
    }
};

template<typename T>
struct dt<1, T>
{
    static T det(const Mat<1, 1, T>& src)
    {
        return src[0][0];
    }
};

/////////////////////////////////////////////////////////////////////////////////

template<size_t DimRows, size_t DimCols, typename T>
class Mat
{
    using braced_list = std::initializer_list<std::initializer_list<T>>;
    using TMat = Mat<DimRows, DimCols, T>;
    using TVec = Vec<DimCols, T>;

    Vec<DimCols, T> rows[DimRows];
public:
    Mat() {}

    Mat(const braced_list& list)
    {
        size_t row_idx = 0;
        for (auto& row: list) {
            size_t col_idx = 0;
            for (auto& elem: row) {
                rows[row_idx][col_idx] = elem;
                col_idx++;
            }
            row_idx++;
        }
    }

    TVec& operator [] (const size_t row_idx)
    {
        assert(row_idx < DimRows);
        return rows[row_idx];
    }

    const TVec& operator [] (const size_t row_idx) const
    {
        assert(row_idx < DimRows);
        return rows[row_idx];
    }

    Vec<DimCols, T> col(const size_t col_idx) const
    {
        assert(col_idx < DimCols);
        Vec<DimCols, T> col;
        for (size_t row_idx = 0; row_idx < DimRows; row_idx++)
        {
            col[row_idx] = 0; //rows[row_idx][col_idx];
        }
        return col;
    }

    void set_col(size_t col_idx, TVec vec)
    {
        assert(col_idx < DimCols);
        for (size_t row_idx = 0; row_idx < DimRows; row_idx++)
        {
            rows[row_idx][col_idx] = vec[row_idx];
        }
    }

    static TMat Identity()
    {
        TMat ret;
        for (size_t i = 0; i < DimRows; i++)
        {
            for (size_t j = 0; j < DimCols; j++)
            {
                ret[i][j] = (i == j) ? 1 : 0;
            }
        }
        return ret;
    }

    T det() const
    {
        return dt<DimCols, T>::det(*this);
    }

    Mat<DimRows - 1, DimCols - 1, T> get_minor(size_t row, size_t col) const
    {
        Mat<DimRows - 1, DimCols - 1, T> ret;
        for (size_t i = 0; i < DimRows - 1; i++)
        {
            for (size_t j = 0; j < DimCols - 1; j++)
            {
                ret[i][j] = rows[i < row ? i : i + 1][j < col ? j : j + 1];
            }
        }
        return ret;
    }

    T cofactor(size_t row, size_t col) const
    {
        return get_minor(row, col).det() * ((row + col) % 2 ? -1 : 1);
    }

    TMat adjugate() const
    {
        TMat ret;
        for (size_t i = 0; i < DimRows; i++)
        {
            for (size_t j = 0; j < DimCols; j++)
            {
                ret[i][j] = cofactor(i, j);
            }
        }
        return ret;
    }

    TMat invert_transpose()
    {
        TMat ret = adjugate();
        T tmp = ret[0] * rows[0];
        return ret / tmp;
    }

    TMat invert()
    {
        return invert_transpose().transpose();
    }

    TMat transpose()
    {
        TMat ret;
        for (size_t i = 0; i < DimCols; )
        {
            ret[i] = this->col(i);
        }
        return ret;
    }

};

/////////////////////////////////////////////////////////////////////////////////

template<size_t DimRows, size_t DimCols, typename T>
Vec<DimRows, T> operator * (const Mat<DimRows, DimCols, T> mat, Vec<DimCols, T> vec)
{
    Vec<DimRows, T> ret;
    for (size_t i = 0; i < DimRows; i++)
    {
        ret[i] = mat[i] * vec;
    }
    return ret;
}

template<size_t R1, size_t C1, size_t C2, typename T>
Mat<R1, C2, T> operator * (Mat<R1, C1, T> l_mat, Mat<C1, C2, T> r_mat)
{
    Mat<R1, C2, T> result;
    for (size_t i = 0; i < R1; i++)
    {
        for (size_t j = 0; j < C2; j++)
        {
            result[i][j] = l_mat[i] * r_mat.col(j);
        }
    }
    return result;
}

template<size_t DimRows, size_t DimCols, typename T>
Mat<DimCols, DimRows, T> operator / (Mat<DimRows, DimCols, T> mat, T scalar)
{
    for (size_t i = 0; i < DimRows; i++)
    {
        mat[i] = mat[i] / scalar;
    }
    return mat;
}

template<size_t DimRows, size_t DimCols, class T>
std::ostream &operator << (std::ostream &out, Mat<DimRows, DimCols, T> &m)
{
    for (size_t i = 0; i < DimRows; i++)
    { out << m[i] << std::endl; }
    return out;
}

/////////////////////////////////////////////////////////////////////////////////

    using Mat4x4 = Mat<4, 4, float>;
    using Mat3x3 = Mat<3, 3, float>;

    /////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////

//template<>
//template<>
//Vec3i::vec(const VecBase<3, float> &v) : x(int(v.x + .5f)), y(int(v.y + .5f)), z(int(v.z + .5f))
//{}
//
//template<>
//template<>
//Vec3::vec(const VecBase<3, int> &v)   : x(v.x), y(v.y), z(v.z)
//{}
//
//template<>
//template<>
//Vec2i::vec(const VecBase<2, float> &v) : x(int(v.x + .5f)), y(int(v.y + .5f))
//{}
//
//template<>
//template<>
//Vec2::vec(const VecBase<2, int> &v)   : x(v.x), y(v.y)
//{}

}




#endif //__GEOMETRY_H__