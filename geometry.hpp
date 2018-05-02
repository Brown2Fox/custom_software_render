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
        virtual T& operator [] (size_t idx) = 0;

    protected METHODS:

        void init()
        {
            for (size_t i = 0; i < DIM; i++)
            {
                (*this)[i] = T();
            }
        }

        void init(const VecBase<DIM,T>& other)
        {
            for (size_t i = 0; i < DIM; i++)
            {
                (*this)[i] = other[i];
            }
        }

        void init(const braced_list& list)
        {
            assert(list.size() == DIM);

            size_t i = 0;
            for (auto& el: list)
            {
                (*this)[i++] = el;
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

        Vec() { TBase::init(); }

        Vec(const VecBase<DIM,T>& other) { TBase::init(other); }

        Vec(const braced_list& list) { TBase::init(list); }

    public OPERATORS:

        T& operator [] (const size_t idx) override
        {
            return components_[idx];
        }

        const T& operator [] (const size_t idx) const
        {
            return components_[idx];
        };


    protected FIELDS:
        T components_[DIM];
    };


/////////////////////////////////////////////////////////////////////////////////


    template <typename T>
    class Vec<2, T>: public VecBase<2, T>
    {
    private:

        static const int DIM = 2;

        using braced_list = std::initializer_list<T>;
        using TClass = Vec<DIM,T>;
        using TBase = VecBase<DIM,T>;

    public FIELDS:
        T x, y;

    public CTORS:

        Vec() { TBase::init(); }

        Vec(const VecBase<DIM,T>& other) { TBase::init(other); }

        Vec(const braced_list& list) { TBase::init(list); }

    public OPERATORS:

        T& operator [] (const size_t idx) override
        {
            return (idx == 0) ? x : (idx == 1) ? y : x;
        }

        const T& operator [] (const size_t idx) const
        {
            return (idx == 0) ? x : (idx == 1) ? y : x;
        };

    };

    using Vec2f = Vec<2, float>;
    using Vec2i = Vec<2, int>;

    template <typename T>
    class Vec<3, T>: public VecBase<3, T>
    {

    private:

        static const int DIM = 3;
        using braced_list = std::initializer_list<T>;
        using TClass = Vec<DIM,T>;
        using TBase = VecBase<DIM,T>;

    public FIELDS:

        T x, y, z;

    public CTORS:

        Vec() { TBase::init(); }

        Vec(const VecBase<DIM,T>& other) { TBase::init(other); }

        Vec(const braced_list& list) { TBase::init(list); }

    public OPERATORS:

        T& operator [] (const size_t idx) override
        {
            return (idx == 0) ? x : (idx == 1) ? y : (idx == 2) ? z : x;
        }

        const T& operator [] (const size_t idx) const
        {
            return (idx == 0) ? x : (idx == 1) ? y : (idx == 2) ? z : x;
        };

    };

    using Vec3f = Vec<3, float>;
    using Vec3i = Vec<3, int>;

    template <typename T>
    class Vec<4, T> : public VecBase<4, T>
    {

    private:

        static const int DIM = 4;
        using braced_list = std::initializer_list<T>;
        using TClass = Vec<DIM,T>;
        using TBase = VecBase<DIM,T>;

    public FIELDS:

        T x, y, z, w;

    public CTORS:

        Vec() { TBase::init(); }

        Vec(const VecBase<DIM,T>& other) { TBase::init(other); }

        Vec(const braced_list& list) { TBase::init(list); }

    public OPERATORS:

        T& operator [] (const size_t idx) override
        {
            return (idx == 0) ? x : (idx == 1) ? y : (idx == 2) ? z : (idx == 3) ? w : x;
        }

        const T& operator [] (const size_t idx) const
        {
            return (idx == 0) ? x : (idx == 1) ? y : (idx == 2) ? z : (idx == 3) ? w : x;
        };
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
    T length(const Vec<DIM, T>& vec)
    {
        T sqared_lenght{0};
        for (int i = 0; i < DIM; i++)
        {
            sqared_lenght += vec[i] * vec[i];
        }
        return std::sqrt(sqared_lenght);
    };


    template <size_t DIM, typename T>
    Vec<DIM, T> normalize(Vec<DIM, T>&& vec, T len = 1)
    {
        Vec<DIM, T> res{};
        auto length = gm::length(vec);

        for (int i = 0; i < DIM; i++)
        {
            res[i] = vec[i] * (len / length); //
        }
        return res;
    }

    template <size_t DIM, typename T>
    Vec<DIM, T> normalize(Vec<DIM, T>& vec, T len = 1)
    {
        Vec<DIM, T> res{0,0,0};
        auto length = gm::length(vec);
        if (length == 1) return vec;

        for (int i = 0; i < DIM; i++)
        {
            res[i] = vec[i] * (len / length);
        }
        return res;
    }

    template <size_t DIM, typename T>
    T dot(const Vec<DIM, T>& vec1, const Vec<DIM, T>& vec2)
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
        return Vec<3, T>{v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x};
    }

    template<typename T>
    Vec<3, T> operator^(Vec<3, T> v1, Vec<3, T> v2)
    {
        return Vec<3, T>{v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x};
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


/////////////////////////////////////////////////////////////////////////////////

    template<size_t DimRows, size_t DimCols, typename T>
    class MatBase
    {
        using braced_list = std::initializer_list<std::initializer_list<T>>;
        using TClass = Mat<DimRows, DimCols, T>;
        using TVec = Vec<DimCols, T>;

        static T det(const MatBase<DimCols, DimCols, T>& src)
        {
            T ret{};
            for (size_t i = 0; i < DimCols; i++)
            {
                ret += src[{0,i}] * src.get_cofactor(0, i);
            }
            return ret;
        }


        static T det(const MatBase<1, 1, T>& src)
        {
            return src[{0,0}];
        }


    protected METHODS:
        virtual T& operator [] (const std::array<size_t, 2>& idxs) = 0;

        virtual TVec get_row(size_t row_idx) const = 0;
        virtual TVec get_col(size_t col_idx) const = 0;
        virtual void set_row(size_t row_idx,  const TVec& row) = 0;
        virtual void set_col(size_t col_idx,  const TVec& col) = 0;

        void init() {}

        void init(const braced_list& list)
        {
            size_t row_idx = 0;
            for (auto& row: list) {
                size_t col_idx = 0;
                for (auto& elem: row) {
                    (*this)[{row_idx, col_idx}] = elem;
                    col_idx++;
                }
                row_idx++;
            }
        }


    public METHODS:

        static TClass Identity()
        {
            TClass ret;
            for (size_t i = 0; i < DimRows; i++)
            {
                for (size_t j = 0; j < DimCols; j++)
                {
                    ret[{i,j}] = (i == j) ? 1 : 0;
                }
            }
            return ret;
        }


        T det() const
        {
            return det(*this);
        }

        Mat<DimRows - 1, DimCols - 1, T> get_minor(size_t row, size_t col) const
        {
            Mat<DimRows - 1, DimCols - 1, T> ret;
            for (size_t i = 0; i < DimRows - 1; i++)
            {
                for (size_t j = 0; j < DimCols - 1; j++)
                {
                    auto ii = (i < row) ? i : i + 1;
                    auto jj = (j < col) ? j : j + 1;
                    ret[{i,j}] = (*this)[{ii, jj}];
                }
            }
            return ret;
        }

        T get_cofactor(size_t row, size_t col) const
        {
            return get_minor(row, col).det() * ((row + col) % 2 ? -1 : 1);
        }

        TClass get_adjugated() const
        {
            TClass ret;
            for (size_t i = 0; i < DimRows; i++)
            {
                for (size_t j = 0; j < DimCols; j++)
                {
                    ret[{i,j}] = get_cofactor(i, j);
                }
            }
            return ret;
        }

        TClass get_invert_transpose()
        {
            TClass ret = get_adjugated();
            T tmp = ret.get_row(0) * this->get_row(0);
            return ret / tmp;
        }

        TClass get_inverted()
        {
            return get_invert_transpose().get_transposed();
        }

        TClass get_transposed()
        {
            TClass ret{};
            for (size_t i = 0; i < DimCols; i++)
            {
                ret.set_row(i, this->get_col(i));
            }
            return ret;
        }


    };

    template<size_t DimRows, size_t DimCols, typename T>
    class Mat: public MatBase<DimRows, DimCols, T>
    {

        using braced_list = std::initializer_list<std::initializer_list<T>>;
        using TClass = Mat<DimRows, DimCols, T>;
        using TVec = Vec<DimCols, T>;
        using TBase = MatBase<DimRows, DimCols, T>;

    public CTORS:

        Mat() { TBase::init(); };
        Mat(const braced_list& list) { TBase::init(list); }

    public METHODS:

        virtual T& operator [] (const std::array<size_t, 2>& idxs) override
        {
            return rows_[idxs[0]][idxs[1]];
        };

        TVec get_row(const size_t row_idx) const override
        {
            return rows_[row_idx];
        }

        TVec get_col(const size_t col_idx) const override
        {
            TVec vec{};
            for (size_t idx = 0; idx < DimCols; idx++) {
                vec[idx] = rows_[idx][col_idx];
            }
            return vec;
        }

        void set_row(const size_t row_idx, const TVec& row) override
        {
            rows_[row_idx] = row;
        }

        void set_col(const size_t col_idx, const TVec& col) override
        {
            for (size_t idx = 0; idx < DimCols; idx++)
            {
                rows_[idx][col_idx] = col[idx];
            }
        }


    private FIELDS:

        TVec rows_[DimRows];
    };



/////////////////////////////////////////////////////////////////////////////////

template<size_t DimRows, size_t DimCols, typename T>
Vec<DimRows, T> operator * (Mat<DimRows, DimCols, T> mat, Vec<DimCols, T> vec)
{
    Vec<DimRows, T> ret;
    for (size_t i = 0; i < DimRows; i++)
    {
        ret[i] = mat.get_row(i) * vec;
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
            result[{i,j}] = l_mat.get_row(i) * r_mat.get_col(j);
        }
    }
    return result;
}

template<size_t DimRows, size_t DimCols, typename T>
Mat<DimCols, DimRows, T> operator / (Mat<DimRows, DimCols, T> mat, T scalar)
{
    for (size_t i = 0; i < DimRows; i++)
    {
        mat.set_row(i, mat.get_col(i) / scalar);
    }
    return mat;
}

template<size_t DimRows, size_t DimCols, class T>
std::ostream &operator << (std::ostream &out, const Mat<DimRows, DimCols, T> &m)
{
    for (size_t i = 0; i < DimRows; i++)
    { out << m.get_row(i) << std::endl; }
    return out;
}

/////////////////////////////////////////////////////////////////////////////////

    using Mat4x4 = Mat<4, 4, float>;
    using Mat3x3 = Mat<3, 3, float>;

    /////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////

//template<>
//template<>
//Vec3i::vec_line(const VecBase<3, float> &v) : x(int(v.x + .5f)), y(int(v.y + .5f)), z(int(v.z + .5f))
//{}
//
//template<>
//template<>
//Vec3::vec_line(const VecBase<3, int> &v)   : x(v.x), y(v.y), z(v.z)
//{}
//
//template<>
//template<>
//Vec2i::vec_line(const VecBase<2, float> &v) : x(int(v.x + .5f)), y(int(v.y + .5f))
//{}
//
//template<>
//template<>
//Vec2::vec_line(const VecBase<2, int> &v)   : x(v.x), y(v.y)
//{}


}

namespace std
{

    template <size_t Dim, typename T>
    void swap(const gm::Vec<Dim, T>& lhs, const gm::Vec<Dim, T> rhs)
    {
        T tmp[Dim]{};

        for (size_t i = 0; i < Dim; i++)
        {
            tmp[i] = lhs[i];
            lhs[i] = rhs[i];
            rhs[i] = tmp[i];
        }
    };

}


#endif //__GEOMETRY_H__