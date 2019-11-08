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

#include <ginkgo/ginkgo.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>


int main(int argc, char *argv[])
{
    using dense = gko::matrix::Dense<>;
    using csr = gko::matrix::Csr<>;
    using coo = gko::matrix::Coo<>;
    using bj = gko::preconditioner::Jacobi<>;

    auto strategy = std::make_shared<csr::automatical>(32);
    // auto strategy = std::make_shared<csr::classical>();

    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    std::shared_ptr<gko::Executor> exec;
    if (argc == 1 || std::string(argv[1]) == "reference") {
        exec = gko::ReferenceExecutor::create();
    } else if (argc == 2 && std::string(argv[1]) == "omp") {
        exec = gko::OmpExecutor::create();
    } else if (argc == 2 && std::string(argv[1]) == "cuda" &&
               gko::CudaExecutor::get_num_devices() > 0) {
        exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    } else if (argc == 2 && std::string(argv[1]) == "hip" &&
               gko::HipExecutor::get_num_devices() > 0) {
        exec = gko::HipExecutor::create(0, gko::OmpExecutor::create());
    } else {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    std::shared_ptr<gko::log::Stream<>> stream_logger =
        gko::log::Stream<>::create(
            exec, gko::log::Logger::iteration_complete_mask, std::cout);

    std::string location_matrices{
        "/home/thomas/Downloads/Matrices_Luca_Azzolin/Reentry"};
    std::string location_Ki = location_matrices + "/Ki_reentries.mtx";
    std::string location_Mi = location_matrices + "/Mi_reentries.mtx";
    std::string location_vm = location_matrices + "/vm_reentry.mtx";
    auto Ki = gko::read<csr>(std::ifstream(location_Ki), exec);
    // auto Ki = csr::create(exec, strategy);
    // Ki->copy_from(lend(Ki_temp));
    auto Mi = gko::share(gko::read<csr>(std::ifstream(location_Mi), exec));
    // auto Mi = share(csr::create(exec, strategy));
    // Mi->copy_from(lend(Mi_temp));
    auto vm = gko::read<dense>(std::ifstream(location_vm), exec);
    auto delta_vm =
        dense::create(exec, gko::dim<2>{Mi->get_size()[0], vm->get_size()[1]});
    auto b = dense::create(
        exec, gko::dim<2>(Ki->get_size()[0], delta_vm->get_size()[1]));

    auto one = gko::initialize<dense>({1.0}, exec);
    auto neg_one = gko::initialize<dense>({-1.0}, exec);
    auto zero = gko::initialize<dense>({0.0}, exec);

    // std::cout << "1. Works" << std::endl;
    Ki->set_strategy(strategy);
    Mi->set_strategy(strategy);

    Ki->apply(lend(vm), lend(b));
    // std::cout << "2. Works" << std::endl;
    b->scale(lend(neg_one));


    auto solver_gen =
        gko::solver::Cg<>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(1000u).on(exec),
                gko::stop::ResidualNormReduction<>::build()
                    .with_reduction_factor(1e-15)
                    .on(exec))
            .with_preconditioner(bj::build().with_max_block_size(32u).on(exec))
            .on(exec);
    auto solver = solver_gen->generate(Mi);
    // solver->add_logger(stream_logger);

    // std::cout << "3. Works" << std::endl;
    auto start_solver = std::chrono::steady_clock::now();
    solver->apply(lend(b), lend(delta_vm));
    auto end_solver = std::chrono::steady_clock::now();
    // std::cout << "4. Works" << std::endl;

    auto dur_solver = std::chrono::duration_cast<std::chrono::microseconds>(
                          end_solver - start_solver)
                          .count();

    std::cout << "Time for solving [us]: " << dur_solver << std::endl;

    // std::cout << "Solution (x): \n";
    // write(std::cout, lend(x));

    auto diff_vector = dense::create_with_config_of(lend(delta_vm));
    auto res = dense::create(exec, gko::dim<2>{1, delta_vm->get_size()[1]});
    Mi->apply(lend(delta_vm), lend(diff_vector));
    diff_vector->add_scaled(lend(neg_one), lend(b));
    diff_vector->compute_norm2(lend(res));

    std::cout << "Residual norm sqrt(r^T r): \n";
    write(std::cout, lend(res));
    //*/
}
