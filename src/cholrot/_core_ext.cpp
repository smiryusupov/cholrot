// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace {

using Array1 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Array2 = py::array_t<double, py::array::c_style | py::array::forcecast>;

inline double get2(const double* a, py::ssize_t ncols, py::ssize_t i, py::ssize_t j) {
    return a[i * ncols + j];
}

inline void set2(double* a, py::ssize_t ncols, py::ssize_t i, py::ssize_t j, double value) {
    a[i * ncols + j] = value;
}

inline void add2(double* a, py::ssize_t ncols, py::ssize_t i, py::ssize_t j, double value) {
    a[i * ncols + j] += value;
}

py::array_t<double> make_array2(py::ssize_t rows, py::ssize_t cols) {
    return py::array_t<double>(std::vector<py::ssize_t>{rows, cols});
}

int parse_method(const std::string& method) {
    if (method == "hy") return 0;
    if (method == "hc") return 1;
    if (method == "algorithm_a" || method == "a" || method == "alga" || method == "alg_a") return 2;
    throw std::invalid_argument("method must be one of {'hy', 'hc', 'algorithm_a'}");
}

void check_alpha(double alpha) {
    if (!(alpha == 1.0 || alpha == -1.0)) {
        throw std::invalid_argument("alpha must be +1 for an update or -1 for a downdate");
    }
}

void check_shapes(const Array2& R, const Array1& z) {
    if (R.ndim() != 2 || R.shape(0) != R.shape(1)) {
        throw std::invalid_argument("R must be a square 2D array");
    }
    if (z.ndim() != 1 || z.shape(0) != R.shape(0)) {
        throw std::invalid_argument("z must be a one-dimensional array with length R.shape[0]");
    }
}

std::vector<double> copy_z(const Array1& z) {
    const double* zp = z.data();
    return std::vector<double>(zp, zp + z.shape(0));
}

py::array_t<double> factor_hy(const Array2& R, const Array1& z_in, double alpha, bool lower) {
    const py::ssize_t n = R.shape(0);
    const double* Rp = R.data();
    std::vector<double> z = copy_z(z_in);
    py::array_t<double> D = make_array2(n, n);
    auto Dbuf = D.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < n; ++i) {
        for (py::ssize_t j = 0; j < n; ++j) Dbuf(i, j) = 0.0;
    }

    for (py::ssize_t i = 0; i < n; ++i) {
        const double rii = get2(Rp, n, i, i);
        const double disc = rii * rii + alpha * z[i] * z[i];
        if (disc <= 0.0 || !std::isfinite(disc)) {
            throw std::domain_error("rank-one operation is not positive definite");
        }
        const double dii = std::sqrt(disc);
        const double ch = rii / dii;
        const double sh = z[i] / dii;
        Dbuf(i, i) = dii;
        if (lower) {
            for (py::ssize_t j = i + 1; j < n; ++j) {
                const double rji = get2(Rp, n, j, i);
                const double dji = ch * rji + alpha * sh * z[j];
                z[j] = -sh * rji + ch * z[j];
                Dbuf(j, i) = dji;
            }
        } else {
            for (py::ssize_t j = i + 1; j < n; ++j) {
                const double rij = get2(Rp, n, i, j);
                const double dij = ch * rij + alpha * sh * z[j];
                z[j] = -sh * rij + ch * z[j];
                Dbuf(i, j) = dij;
            }
        }
    }
    return D;
}

py::array_t<double> factor_hc(const Array2& R, const Array1& z_in, double alpha, bool lower) {
    const py::ssize_t n = R.shape(0);
    const double* Rp = R.data();
    std::vector<double> z = copy_z(z_in);
    py::array_t<double> D = make_array2(n, n);
    auto Dbuf = D.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < n; ++i) {
        for (py::ssize_t j = 0; j < n; ++j) Dbuf(i, j) = 0.0;
    }

    for (py::ssize_t i = 0; i < n; ++i) {
        const double rii = get2(Rp, n, i, i);
        const double s = z[i] / rii;
        const double c2 = 1.0 + alpha * s * s;
        if (c2 <= 0.0 || !std::isfinite(c2)) {
            throw std::domain_error("rank-one operation is not positive definite");
        }
        const double c = std::sqrt(c2);
        Dbuf(i, i) = c * rii;
        if (lower) {
            for (py::ssize_t j = i + 1; j < n; ++j) {
                const double rji = get2(Rp, n, j, i);
                const double dji = (rji + alpha * s * z[j]) / c;
                z[j] = -s * dji + c * z[j];
                Dbuf(j, i) = dji;
            }
        } else {
            for (py::ssize_t j = i + 1; j < n; ++j) {
                const double rij = get2(Rp, n, i, j);
                const double dij = (rij + alpha * s * z[j]) / c;
                z[j] = -s * dij + c * z[j];
                Dbuf(i, j) = dij;
            }
        }
    }
    return D;
}

py::array_t<double> factor_algorithm_a(const Array2& R, const Array1& z_in, double alpha, bool lower) {
    const py::ssize_t n = R.shape(0);
    const double* Rp = R.data();
    std::vector<double> z = copy_z(z_in);
    py::array_t<double> D = make_array2(n, n);
    auto Dbuf = D.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < n; ++i) {
        for (py::ssize_t j = 0; j < n; ++j) Dbuf(i, j) = 0.0;
    }

    double lam_prev = 1.0;
    for (py::ssize_t i = 0; i < n; ++i) {
        const double rii = get2(Rp, n, i, i);
        const double a_i = z[i] / rii;
        const double lam_next = lam_prev + alpha * a_i * a_i;
        if (lam_next <= 0.0 || !std::isfinite(lam_next)) {
            throw std::domain_error("rank-one operation is not positive definite");
        }
        const double beta = std::sqrt(lam_next);
        const double beta_prev = std::sqrt(lam_prev);
        const double scale = beta / beta_prev;
        Dbuf(i, i) = scale * rii;
        if (lower) {
            for (py::ssize_t j = i + 1; j < n; ++j) {
                const double rji = get2(Rp, n, j, i);
                z[j] = z[j] - a_i * rji;
                const double dji = scale * rji + alpha * (a_i / (beta_prev * beta)) * z[j];
                Dbuf(j, i) = dji;
            }
        } else {
            for (py::ssize_t j = i + 1; j < n; ++j) {
                const double rij = get2(Rp, n, i, j);
                z[j] = z[j] - a_i * rij;
                const double dij = scale * rij + alpha * (a_i / (beta_prev * beta)) * z[j];
                Dbuf(i, j) = dij;
            }
        }
        lam_prev = lam_next;
    }
    return D;
}

py::array_t<double> matvec_hy(const Array2& R, const Array1& z_in, const Array2& V, double alpha, bool lower) {
    const py::ssize_t n = R.shape(0);
    const py::ssize_t k = V.shape(1);
    const double* Rp = R.data();
    const double* Vp = V.data();
    std::vector<double> z = copy_z(z_in);
    py::array_t<double> W = make_array2(n, k);
    auto Wbuf = W.mutable_unchecked<2>();
    double* Wp = W.mutable_data();
    for (py::ssize_t i = 0; i < n; ++i) {
        for (py::ssize_t p = 0; p < k; ++p) Wbuf(i, p) = 0.0;
    }

    for (py::ssize_t i = 0; i < n; ++i) {
        const double rii = get2(Rp, n, i, i);
        const double disc = rii * rii + alpha * z[i] * z[i];
        if (disc <= 0.0 || !std::isfinite(disc)) {
            throw std::domain_error("rank-one operation is not positive definite");
        }
        const double dii = std::sqrt(disc);
        const double ch = rii / dii;
        const double sh = z[i] / dii;
        if (lower) {
            for (py::ssize_t p = 0; p < k; ++p) add2(Wp, k, i, p, dii * get2(Vp, k, i, p));
            for (py::ssize_t j = i + 1; j < n; ++j) {
                const double rji = get2(Rp, n, j, i);
                const double dji = ch * rji + alpha * sh * z[j];
                z[j] = -sh * rji + ch * z[j];
                for (py::ssize_t p = 0; p < k; ++p) add2(Wp, k, j, p, dji * get2(Vp, k, i, p));
            }
        } else {
            for (py::ssize_t p = 0; p < k; ++p) add2(Wp, k, i, p, dii * get2(Vp, k, i, p));
            for (py::ssize_t j = i + 1; j < n; ++j) {
                const double rij = get2(Rp, n, i, j);
                const double dij = ch * rij + alpha * sh * z[j];
                z[j] = -sh * rij + ch * z[j];
                for (py::ssize_t p = 0; p < k; ++p) add2(Wp, k, i, p, dij * get2(Vp, k, j, p));
            }
        }
    }
    return W;
}

py::array_t<double> matvec_hc(const Array2& R, const Array1& z_in, const Array2& V, double alpha, bool lower) {
    const py::ssize_t n = R.shape(0);
    const py::ssize_t k = V.shape(1);
    const double* Rp = R.data();
    const double* Vp = V.data();
    std::vector<double> z = copy_z(z_in);
    py::array_t<double> W = make_array2(n, k);
    auto Wbuf = W.mutable_unchecked<2>();
    double* Wp = W.mutable_data();
    for (py::ssize_t i = 0; i < n; ++i) {
        for (py::ssize_t p = 0; p < k; ++p) Wbuf(i, p) = 0.0;
    }

    for (py::ssize_t i = 0; i < n; ++i) {
        const double rii = get2(Rp, n, i, i);
        const double s = z[i] / rii;
        const double c2 = 1.0 + alpha * s * s;
        if (c2 <= 0.0 || !std::isfinite(c2)) {
            throw std::domain_error("rank-one operation is not positive definite");
        }
        const double c = std::sqrt(c2);
        const double dii = c * rii;
        if (lower) {
            for (py::ssize_t p = 0; p < k; ++p) add2(Wp, k, i, p, dii * get2(Vp, k, i, p));
            for (py::ssize_t j = i + 1; j < n; ++j) {
                const double rji = get2(Rp, n, j, i);
                const double dji = (rji + alpha * s * z[j]) / c;
                z[j] = -s * dji + c * z[j];
                for (py::ssize_t p = 0; p < k; ++p) add2(Wp, k, j, p, dji * get2(Vp, k, i, p));
            }
        } else {
            for (py::ssize_t p = 0; p < k; ++p) add2(Wp, k, i, p, dii * get2(Vp, k, i, p));
            for (py::ssize_t j = i + 1; j < n; ++j) {
                const double rij = get2(Rp, n, i, j);
                const double dij = (rij + alpha * s * z[j]) / c;
                z[j] = -s * dij + c * z[j];
                for (py::ssize_t p = 0; p < k; ++p) add2(Wp, k, i, p, dij * get2(Vp, k, j, p));
            }
        }
    }
    return W;
}

py::array_t<double> matvec_algorithm_a(const Array2& R, const Array1& z_in, const Array2& V, double alpha, bool lower) {
    const py::ssize_t n = R.shape(0);
    const py::ssize_t k = V.shape(1);
    const double* Rp = R.data();
    const double* Vp = V.data();
    std::vector<double> z = copy_z(z_in);
    py::array_t<double> W = make_array2(n, k);
    auto Wbuf = W.mutable_unchecked<2>();
    double* Wp = W.mutable_data();
    for (py::ssize_t i = 0; i < n; ++i) {
        for (py::ssize_t p = 0; p < k; ++p) Wbuf(i, p) = 0.0;
    }

    double lam_prev = 1.0;
    for (py::ssize_t i = 0; i < n; ++i) {
        const double rii = get2(Rp, n, i, i);
        const double a_i = z[i] / rii;
        const double lam_next = lam_prev + alpha * a_i * a_i;
        if (lam_next <= 0.0 || !std::isfinite(lam_next)) {
            throw std::domain_error("rank-one operation is not positive definite");
        }
        const double beta = std::sqrt(lam_next);
        const double beta_prev = std::sqrt(lam_prev);
        const double scale = beta / beta_prev;
        const double dii = scale * rii;
        if (lower) {
            for (py::ssize_t p = 0; p < k; ++p) add2(Wp, k, i, p, dii * get2(Vp, k, i, p));
            for (py::ssize_t j = i + 1; j < n; ++j) {
                const double rji = get2(Rp, n, j, i);
                z[j] = z[j] - a_i * rji;
                const double dji = scale * rji + alpha * (a_i / (beta_prev * beta)) * z[j];
                for (py::ssize_t p = 0; p < k; ++p) add2(Wp, k, j, p, dji * get2(Vp, k, i, p));
            }
        } else {
            for (py::ssize_t p = 0; p < k; ++p) add2(Wp, k, i, p, dii * get2(Vp, k, i, p));
            for (py::ssize_t j = i + 1; j < n; ++j) {
                const double rij = get2(Rp, n, i, j);
                z[j] = z[j] - a_i * rij;
                const double dij = scale * rij + alpha * (a_i / (beta_prev * beta)) * z[j];
                for (py::ssize_t p = 0; p < k; ++p) add2(Wp, k, i, p, dij * get2(Vp, k, j, p));
            }
        }
        lam_prev = lam_next;
    }
    return W;
}



py::array_t<double> identity_matvec_c_impl(const Array1& z_in, const Array2& V, double alpha, bool lower) {
    const py::ssize_t n = z_in.shape(0);
    const py::ssize_t k = V.shape(1);
    const double* zp = z_in.data();
    const double* Vp = V.data();
    py::array_t<double> W = make_array2(n, k);
    double* Wp = W.mutable_data();

    double q = 1.0;
    if (lower) {
        std::vector<double> prefix(static_cast<std::size_t>(k), 0.0);
        for (py::ssize_t i = 0; i < n; ++i) {
            const double zi = zp[i];
            const double q_next = q + alpha * zi * zi;
            if (q_next <= 0.0 || !std::isfinite(q_next)) {
                throw std::domain_error("rank-one operation is not positive definite");
            }
            const double diag = std::sqrt(q_next / q);
            const double denom = std::sqrt(q * q_next);
            for (py::ssize_t p = 0; p < k; ++p) {
                const double vip = get2(Vp, k, i, p);
                set2(Wp, k, i, p, diag * vip + alpha * zi * prefix[static_cast<std::size_t>(p)]);
                prefix[static_cast<std::size_t>(p)] += zi * vip / denom;
            }
            q = q_next;
        }
    } else {
        std::vector<double> suffix(static_cast<std::size_t>(k), 0.0);
        for (py::ssize_t i = 0; i < n; ++i) {
            const double zi = zp[i];
            for (py::ssize_t p = 0; p < k; ++p) {
                suffix[static_cast<std::size_t>(p)] += zi * get2(Vp, k, i, p);
            }
        }
        for (py::ssize_t i = 0; i < n; ++i) {
            const double zi = zp[i];
            for (py::ssize_t p = 0; p < k; ++p) {
                suffix[static_cast<std::size_t>(p)] -= zi * get2(Vp, k, i, p);
            }
            const double q_next = q + alpha * zi * zi;
            if (q_next <= 0.0 || !std::isfinite(q_next)) {
                throw std::domain_error("rank-one operation is not positive definite");
            }
            const double diag = std::sqrt(q_next / q);
            const double denom = std::sqrt(q * q_next);
            for (py::ssize_t p = 0; p < k; ++p) {
                const double vip = get2(Vp, k, i, p);
                set2(Wp, k, i, p, diag * vip + alpha * zi * suffix[static_cast<std::size_t>(p)] / denom);
            }
            q = q_next;
        }
    }
    return W;
}

std::vector<double> triangular_solve_vec(const double* T, const double* B, py::ssize_t n, py::ssize_t k, bool lower, bool trans) {
    std::vector<double> X(static_cast<std::size_t>(n * k), 0.0);
    auto bget = [&](py::ssize_t i, py::ssize_t p) { return B[i * k + p]; };
    auto xget = [&](py::ssize_t i, py::ssize_t p) { return X[static_cast<std::size_t>(i * k + p)]; };
    auto xset = [&](py::ssize_t i, py::ssize_t p, double value) { X[static_cast<std::size_t>(i * k + p)] = value; };

    if (!trans && lower) {
        for (py::ssize_t i = 0; i < n; ++i) {
            for (py::ssize_t p = 0; p < k; ++p) {
                double rhs = bget(i, p);
                for (py::ssize_t j = 0; j < i; ++j) rhs -= get2(T, n, i, j) * xget(j, p);
                xset(i, p, rhs / get2(T, n, i, i));
            }
        }
    } else if (!trans && !lower) {
        for (py::ssize_t ii = 0; ii < n; ++ii) {
            const py::ssize_t i = n - 1 - ii;
            for (py::ssize_t p = 0; p < k; ++p) {
                double rhs = bget(i, p);
                for (py::ssize_t j = i + 1; j < n; ++j) rhs -= get2(T, n, i, j) * xget(j, p);
                xset(i, p, rhs / get2(T, n, i, i));
            }
        }
    } else if (trans && lower) {
        for (py::ssize_t ii = 0; ii < n; ++ii) {
            const py::ssize_t i = n - 1 - ii;
            for (py::ssize_t p = 0; p < k; ++p) {
                double rhs = bget(i, p);
                for (py::ssize_t j = i + 1; j < n; ++j) rhs -= get2(T, n, j, i) * xget(j, p);
                xset(i, p, rhs / get2(T, n, i, i));
            }
        }
    } else {  // trans && !lower
        for (py::ssize_t i = 0; i < n; ++i) {
            for (py::ssize_t p = 0; p < k; ++p) {
                double rhs = bget(i, p);
                for (py::ssize_t j = 0; j < i; ++j) rhs -= get2(T, n, j, i) * xget(j, p);
                xset(i, p, rhs / get2(T, n, i, i));
            }
        }
    }
    return X;
}

std::vector<double> spd_cholesky_solve_vec(const Array2& R, const double* B, py::ssize_t k, bool lower) {
    const py::ssize_t n = R.shape(0);
    const double* Rp = R.data();
    if (lower) {
        std::vector<double> y = triangular_solve_vec(Rp, B, n, k, true, false);
        return triangular_solve_vec(Rp, y.data(), n, k, true, true);
    }
    std::vector<double> y = triangular_solve_vec(Rp, B, n, k, false, true);
    return triangular_solve_vec(Rp, y.data(), n, k, false, false);
}

py::array_t<double> solve_rank1_c_impl(const Array2& R, const Array1& z_in, const Array2& B, double alpha, bool lower) {
    const py::ssize_t n = R.shape(0);
    const py::ssize_t k = B.shape(1);
    const double* Bp = B.data();
    std::vector<double> z = copy_z(z_in);
    std::vector<double> Ainv_b = spd_cholesky_solve_vec(R, Bp, k, lower);
    py::array_t<double> z_array = make_array2(n, 1);
    auto zbuf = z_array.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < n; ++i) zbuf(i, 0) = z[i];
    std::vector<double> Ainv_z = spd_cholesky_solve_vec(R, z_array.data(), 1, lower);

    double zAinv_z = 0.0;
    for (py::ssize_t i = 0; i < n; ++i) zAinv_z += z[i] * Ainv_z[static_cast<std::size_t>(i)];
    const double denom = 1.0 + alpha * zAinv_z;
    if (denom <= 0.0 || !std::isfinite(denom)) {
        throw std::domain_error("rank-one downdate/update is not positive definite");
    }

    std::vector<double> zAinv_b(static_cast<std::size_t>(k), 0.0);
    for (py::ssize_t p = 0; p < k; ++p) {
        for (py::ssize_t i = 0; i < n; ++i) {
            zAinv_b[static_cast<std::size_t>(p)] += z[i] * Ainv_b[static_cast<std::size_t>(i * k + p)];
        }
    }

    py::array_t<double> X = make_array2(n, k);
    auto Xbuf = X.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < n; ++i) {
        for (py::ssize_t p = 0; p < k; ++p) {
            Xbuf(i, p) = Ainv_b[static_cast<std::size_t>(i * k + p)]
                       - alpha * Ainv_z[static_cast<std::size_t>(i)] * (zAinv_b[static_cast<std::size_t>(p)] / denom);
        }
    }
    return X;
}

}  // namespace

py::array_t<double> update_c(const Array2& R, const Array1& z, double alpha, bool lower, const std::string& method) {
    check_alpha(alpha);
    check_shapes(R, z);
    const int m = parse_method(method);
    if (m == 0) return factor_hy(R, z, alpha, lower);
    if (m == 1) return factor_hc(R, z, alpha, lower);
    return factor_algorithm_a(R, z, alpha, lower);
}

py::array_t<double> matvec_c(const Array2& R, const Array1& z, const Array2& V, double alpha, bool lower, const std::string& method) {
    check_alpha(alpha);
    check_shapes(R, z);
    if (V.ndim() != 2 || V.shape(0) != R.shape(0)) {
        throw std::invalid_argument("V must be a 2D array with shape (n, k)");
    }
    const int m = parse_method(method);
    if (m == 0) return matvec_hy(R, z, V, alpha, lower);
    if (m == 1) return matvec_hc(R, z, V, alpha, lower);
    return matvec_algorithm_a(R, z, V, alpha, lower);
}



py::array_t<double> identity_matvec_c(const Array1& z, const Array2& V, double alpha, bool lower) {
    check_alpha(alpha);
    if (z.ndim() != 1) {
        throw std::invalid_argument("z must be a one-dimensional array");
    }
    if (V.ndim() != 2 || V.shape(0) != z.shape(0)) {
        throw std::invalid_argument("V must be a 2D array with shape (n, k)");
    }
    return identity_matvec_c_impl(z, V, alpha, lower);
}

py::array_t<double> solve_rank1_c(const Array2& R, const Array1& z, const Array2& B, double alpha, bool lower) {
    check_alpha(alpha);
    check_shapes(R, z);
    if (B.ndim() != 2 || B.shape(0) != R.shape(0)) {
        throw std::invalid_argument("B must be a 2D array with shape (n, k)");
    }
    return solve_rank1_c_impl(R, z, B, alpha, lower);
}

PYBIND11_MODULE(_core_ext, m) {
    m.doc() = "C++/pybind11 kernels for cholrot";
    m.def("update_c", &update_c, py::arg("R"), py::arg("z"), py::arg("alpha"), py::arg("lower"), py::arg("method"),
          "Materialize a rank-one modified Cholesky factor.");
    m.def("matvec_c", &matvec_c, py::arg("R"), py::arg("z"), py::arg("V"), py::arg("alpha"), py::arg("lower"), py::arg("method"),
          "Compute D @ V directly for a rank-one modified Cholesky factor.");
    m.def("identity_matvec_c", &identity_matvec_c, py::arg("z"), py::arg("V"), py::arg("alpha"), py::arg("lower"),
          "Compute D @ V for D D.T = I + alpha*z*z.T in linear time.");
    m.def("solve_rank1_c", &solve_rank1_c, py::arg("R"), py::arg("z"), py::arg("B"), py::arg("alpha"), py::arg("lower"),
          "Solve (A + alpha*z*z.T) X = B from a base Cholesky factor.");
}
