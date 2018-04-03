#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include <cmath>
#include <vector>
#include <cassert>
#include <iostream>
#include <functional>
#include <ostream>

namespace gm
{


template<size_t DimCols, size_t DimRows, typename T>
class MatBase;




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

    public:
        VecBase()
        {
            for (int i = 0; i < DIM; i++)
            {
                components_[i] = T();
            }
        }

        T& operator [] (const int i)
        {
            assert(i < DIM);
            return components_[i];
        }


        VecBase(std::initializer_list<T> list) {
            assert(list.size() != DIM);
            std::cout << "{list}\n";
            int i = 0;
            for (auto& el: list)
            {
                this->operator[](i++) = el;
            }
        }


//        T magnitude() {
//            T res = T();
//            for (int i = 0; i < DIM; i++)
//            {
//                res += this->operator[](i) * this->operator[](i);
//            }
//
//            return std::sqrt(res);
//        }

    protected:
        T components_[DIM];
    };


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
    T operator * (VecBase<DIM, T> lhs, VecBase<DIM, T> rhs)
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
    VecBase<DIM, T> operator + (VecBase<DIM, T> vec_l, VecBase<DIM, U> vec_r)
    {
        for (size_t i = 0; i < DIM; i++)
        {
            vec_l[i] = vec_l[i] + vec_r[i];
        }
        return vec_l;
    }

    template<size_t DIM, typename T, typename U>
    VecBase<DIM, T> operator - (VecBase<DIM, T> vec_l, VecBase<DIM, U> vec_r)
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
    VecBase<DIM, T> operator * (VecBase<DIM, T> vec, U scalar)
    {
        for (size_t i = DIM; i--;)
        {
            vec[i] = vec[i] * scalar;
        }
        return vec;
    }

    template<size_t DIM, typename T, typename U>
    VecBase<DIM, T> operator / (VecBase<DIM, T> vec, U scalar)
    {
        for (size_t i = DIM; i--;)
        {
            vec[i] = vec[i] / scalar;
        }
        return vec;
    }

    template <size_t DIM, typename T>
    VecBase<DIM, T> normalize(VecBase<DIM, T>&& vec, T len = 1)
    {
        auto res = VecBase<DIM, T>();

        for (int i = 0; i < DIM; i++)
        {
            res[i] = vec[i] * (len / 0.3);
        }
        return res;
    }

    template <size_t DIM, typename T>
    VecBase<DIM, T> normalize(VecBase<DIM, T>& vec, T len = 1)
    {
        auto res = VecBase<DIM, T>();

        for (int i = 0; i < DIM; i++)
        {
            res[i] = vec[i] * (len / 0.3);
        }
        return res;
    }

    template <size_t DIM, typename T>
    T dot(VecBase<DIM, T>& vec1, VecBase<DIM, T>& vec2)
    {
        T result = T();
        for (size_t i = 0; i < DIM; i++)
        {
            result += vec1[i] * vec2[i];
        }
        return result;
    }


    template<size_t LEN, size_t DIM, typename T>
    VecBase<LEN, T> embed(const VecBase<DIM, T> &v, T fill = 1)
    {
        VecBase<LEN, T> ret;
        for (size_t i = LEN; i--;)
        {
            ret[i] = (i < DIM ? v[i] : fill);
        }
        return ret;
    }

    template<size_t LEN, size_t DIM, typename T>
    VecBase<LEN, T> proj(const VecBase<DIM, T> &v)
    {
        VecBase<LEN, T> ret;
        for (size_t i = LEN; i--; ret[i] = v[i])
        {}
        return ret;
    }

    template<typename T>
    VecBase<3, T> cross(VecBase<3, T> v1, VecBase<3, T> v2)
    {
        return VecBase<3, T>(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
    }

    template<typename T>
    VecBase<3, T> operator^(VecBase<3, T> v1, VecBase<3, T> v2)
    {
        return VecBase<3, T>(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
    }


    template <size_t DIM, typename T>
    std::ostream& operator<<(std::ostream& os, const VecBase<DIM, T>& vec)
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
    static T det(const MatBase<DIM, DIM, T> &src)
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
    static T det(const MatBase<1, 1, T> &src)
    {
        return src[0][0];
    }
};

/////////////////////////////////////////////////////////////////////////////////

template<size_t DimRows, size_t DimCols, typename T>
class MatBase
{
    using TMat = MatBase<DimRows, DimCols, T>;
    using TVec = VecBase<DimCols, T>;

    TVec rows[DimRows];
public:
    MatBase() {}

    MatBase(std::initializer_list< TVec >&& list)
    {

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

    TVec col(const size_t col_idx) const
    {
        assert(col_idx < DimCols);
        TVec col;
        for (size_t row_idx = 0; row_idx < DimRows; row_idx++)
        {
            col[row_idx] = rows[row_idx][col_idx];
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

    MatBase<DimRows - 1, DimCols - 1, T> get_minor(size_t row, size_t col) const
    {
        MatBase<DimRows - 1, DimCols - 1, T> ret;
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
VecBase<DimRows, T> operator * (const MatBase<DimRows, DimCols, T> mat, VecBase<DimCols, T> vec)
{
    VecBase<DimRows, T> ret;
    for (size_t i = 0; i < DimRows; i++)
    {
        ret[i] = mat[i] * vec;
    }
    return ret;
}

template<size_t R1, size_t C1, size_t C2, typename T>
MatBase<R1, C2, T> operator * (MatBase<R1, C1, T> l_mat, MatBase<C1, C2, T> r_mat)
{
    MatBase<R1, C2, T> result;
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
MatBase<DimCols, DimRows, T> operator / (MatBase<DimRows, DimCols, T> mat, T scalar)
{
    for (size_t i = 0; i < DimRows; i++)
    {
        mat[i] = mat[i] / scalar;
    }
    return mat;
}

template<size_t DimRows, size_t DimCols, class T>
std::ostream &operator << (std::ostream &out, MatBase<DimRows, DimCols, T> &m)
{
    for (size_t i = 0; i < DimRows; i++)
    { out << m[i] << std::endl; }
    return out;
}

/////////////////////////////////////////////////////////////////////////////////

    using Mat4x4 = MatBase<4, 4, float>;
    using Mat3x3 = MatBase<3, 3, float>;

    /////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    class VecBase<2, T>
    {

    public:

        T& operator [] (const int idx)
        {
            assert(idx < 2);
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

        const T& operator [] (const int idx) const
        {
            assert(idx < 2);
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

        T x, y;
    };

    typedef VecBase<2, float> Vec2f;
    typedef VecBase<2, int> Vec2i;

    template <typename T>
    class VecBase<3, T>
    {
    public:
        T& operator [] (const int idx)
        {
            assert(idx < 3);
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

        const T& operator [] (const int idx) const
        {
            assert(idx < 3);
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

        static VecBase<3, T> Zero() {
            return VecBase<3, T>{0,0,0};
        };


        T x, y, z;
    };

    using Vec3f = VecBase<3, float>;
    using Vec3i = VecBase<3, int>;

    template <typename T>
    class VecBase<4, T>
    {
    public:

        T& operator [] (const int idx)
        {
            assert(idx < 4);
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

        const T& operator [] (const int idx) const
        {
            assert(idx < 4);
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

        T x, y, z, w;
    };

    using Vec4f = VecBase<4, float>;
    using Vec4i = VecBase<4, int>;

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