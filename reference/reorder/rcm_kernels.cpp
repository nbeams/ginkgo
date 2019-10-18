/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/


#include "core/reorder/rcm_kernels.hpp"


#include <bits/stdc++.h>
#include <algorithm>
#include <iterator>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/metis_types.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The metis fill reduce namespace.
 *
 * @ingroup reorder
 */
namespace rcm {


template <typename ValueType, typename IndexType>
void get_degree_of_nodes(
    std::shared_ptr<const ReferenceExecutor> exec,
    std::shared_ptr<matrix::SparsityCsr<ValueType, IndexType>> adjacency_matrix,
    std::shared_ptr<gko::Array<IndexType>> node_degrees)
{
    auto num_rows = adjacency_matrix->get_size()[0];
    auto adj_ptrs = adjacency_matrix->get_row_ptrs();
    auto node_deg = node_degrees->get_data();

    for (auto i = 0; i < num_rows; ++i) {
        node_deg[i] = adj_ptrs[i + 1] - adj_ptrs[i];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_RCM_GET_DEGREE_OF_NODES_KERNEL);


template <typename IndexType>
IndexType findIndex(std::vector<std::pair<IndexType, IndexType>> &a,
                    IndexType x)
{
    for (auto i = 0; i < a.size(); i++)
        if (a[i].first == x) return i;
    return -1;
}


// template <typename IndexType>
// bool compareDegree(int i, int j)
// {
//   return ::globalDegree[i] - ::globalDegree[j];
// }


template <typename ValueType, typename IndexType>
void get_permutation(
    std::shared_ptr<const ReferenceExecutor> exec, size_type num_vertices,
    std::shared_ptr<matrix::SparsityCsr<ValueType, IndexType>> adjacency_matrix,
    std::shared_ptr<Array<IndexType>> node_degrees,
    std::shared_ptr<matrix::Permutation<IndexType>> permutation_mat,
    std::shared_ptr<matrix::Permutation<IndexType>> inv_permutation_mat)
{
    IndexType num_vtxs = static_cast<IndexType>(num_vertices);
    auto adj_ptrs = adjacency_matrix->get_row_ptrs();
    auto adj_idxs = adjacency_matrix->get_col_idxs();
    auto node_deg = node_degrees->get_data();
    auto permutation_arr = permutation_mat->get_permutation();
    auto inv_permutation_arr = inv_permutation_mat->get_permutation();

    std::queue<IndexType> q;
    std::vector<IndexType> r;
    std::vector<std::pair<IndexType, IndexType>> not_visited;

    std::cout << " Here " << __LINE__ << std::endl;
    for (auto i = 0; i < num_vtxs; ++i) {
        not_visited.push_back(std::make_pair(i, node_deg[i]));
    }

    std::cout << " Here " << __LINE__ << std::endl;
    while (not_visited.size()) {
        IndexType minNodeIndex = 0;

        for (auto i = 0; i < not_visited.size(); i++) {
            if (not_visited[i].second < not_visited[minNodeIndex].second) {
                minNodeIndex = i;
            }
        }
        q.push(not_visited[minNodeIndex].first);

        not_visited.erase(not_visited.begin() +
                          findIndex(not_visited, not_visited[q.front()].first));

        std::cout << " Here " << __LINE__ << " q size " << q.size()
                  << std::endl;
        // Simple BFS
        while (!q.empty()) {
            std::vector<IndexType> toSort;

            for (IndexType i = 0; i < num_vtxs; i++) {
                if (i != q.front() &&
                    ((q.front() * num_vtxs + i) == (adj_ptrs[i] + i)) &&
                    findIndex(not_visited, i) != -1) {
                    toSort.push_back(i);
                    not_visited.erase(not_visited.begin() +
                                      findIndex(not_visited, i));
                }
            }

            std::sort(toSort.begin(), toSort.end(), [&node_deg](int i, int j) {
                return node_deg[j] - node_deg[i];
            });

            for (auto i = 0; i < toSort.size(); i++) q.push(toSort[i]);

            r.push_back(q.front());
            q.pop();
        }
    }

    std::cout << " Here " << __LINE__ << " r size " << r.size() << std::endl;
    for (auto i = 0; i < r.size(); ++i) {
        std::cout << " r at " << i << " : " << r[i] << "\n";
        permutation_arr[i] = r[i];
    }
    std::cout << " ] r " << std::endl;
    // std::vector<IndexType> tmp(num_vertices, 0);
    // std::iota(tmp.begin(), tmp.end(), 0);
    // for (auto i = 0; i < num_vertices; ++i) {
    //   permutation_mat->get_permutation()[i] = tmp[i];
    // }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_RCM_GET_PERMUTATION_KERNEL);


}  // namespace rcm
}  // namespace reference
}  // namespace kernels
}  // namespace gko
