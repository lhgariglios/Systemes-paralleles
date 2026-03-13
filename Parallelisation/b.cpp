#include <vector>
#include <iostream>
#include <random>
#include "fractal_land.hpp"
#include <mpi.h>
#include <omp.h>
#include "ant.hpp"
#include "pheronome.hpp"
# include "renderer.hpp"
# include "window.hpp"
# include "rand_generator.hpp"

#include <chrono>

void advance_time( const fractal_land& land, pheronome& phen, 
                   const position_t& pos_nest, const position_t& pos_food,
                   std::vector<int>& pos_x, std::vector<int>& pos_y,
                   std::vector<char>& loaded, std::vector<uint32_t>& seeds, std::size_t& cpteur )
{
    for ( size_t i = 0; i < pos_x.size(); ++i ){
        int& my_pos_x   = pos_x[i];
        int& my_pos_y = pos_y[i];
        char& load  = loaded[i];
        std::size_t my_sd = seeds[i];
        
        auto ant_choice = [&my_sd]() mutable { return rand_double( 0., 1., my_sd ); };
        auto dir_choice = [&my_sd]() mutable { return rand_int32( 1, 4, my_sd ); };
        double consumed_time = 0.;
        const double eps = 0.8; // Coefficiente de exploração
        
        // Tant que la fourmi peut encore bouger dans le pas de temps imparti
        while ( consumed_time < 1. ) {
            // Si la fourmi est chargée, elle suit les phéromones de deuxième type, sinon ceux du premier.
            int        ind_pher    = ( load ? 1 : 0 );
            double     choix       = ant_choice( );
            position_t old_pos_ant = {my_pos_x, my_pos_y};
            position_t new_pos_ant = old_pos_ant;
            double max_phen    = std::max( {phen( new_pos_ant.x - 1, new_pos_ant.y )[ind_pher],
                                         phen( new_pos_ant.x + 1, new_pos_ant.y )[ind_pher],
                                         phen( new_pos_ant.x, new_pos_ant.y - 1 )[ind_pher],
                                         phen( new_pos_ant.x, new_pos_ant.y + 1 )[ind_pher]} );
            if ( ( choix > eps ) || ( max_phen <= 0. ) ) {
                do {
                    new_pos_ant = old_pos_ant;
                    int d = dir_choice();
                    if ( d==1 ) new_pos_ant.x  -= 1;
                    if ( d==2 ) new_pos_ant.y -= 1;
                    if ( d==3 ) new_pos_ant.x  += 1;
                    if ( d==4 ) new_pos_ant.y += 1;

                } while ( phen[new_pos_ant][ind_pher] == -1 );
            } else {
                // On choisit la case où le phéromone est le plus fort.
                if ( phen( new_pos_ant.x - 1, new_pos_ant.y )[ind_pher] == max_phen )
                    new_pos_ant.x -= 1;
                else if ( phen( new_pos_ant.x + 1, new_pos_ant.y )[ind_pher] == max_phen )
                    new_pos_ant.x += 1;
                else if ( phen( new_pos_ant.x, new_pos_ant.y - 1 )[ind_pher] == max_phen )
                    new_pos_ant.y -= 1;
                else  // if (phen(new_pos_ant.first,new_pos_ant.second+1)[ind_pher] == max_phen)
                    new_pos_ant.y += 1;
            }
            consumed_time += land( new_pos_ant.x, new_pos_ant.y);
            phen.mark_pheronome( new_pos_ant );
            my_pos_x = new_pos_ant.x;
            my_pos_y = new_pos_ant.y;
            if ( new_pos_ant == pos_nest ) {
                if ( load ) {
                    cpteur += 1;
                }
                load = 0;
            }
            if ( new_pos_ant == pos_food ) {
                load = 1;
            }
        }
        seeds[i] = my_sd;
    }
        
    phen.do_evaporation();
    phen.update();
}

int main(int argc, char* argv[])
{

    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) SDL_Init( SDL_INIT_VIDEO );

    std::size_t n_iters = 5000;
    std::size_t seed = 2026 + rank; // Graine pour la génération aléatoire ( reproductible )
    const int nb_ants = 5000; // Nombre de fourmis
    const double eps = 0.8;  // Coefficient d'exploration
    const double alpha=0.7; // Coefficient de chaos

    //const double beta=0.9999; // Coefficient d'évaporation
    const double beta=0.999; // Coefficient d'évaporation
    // Location du nid
    position_t pos_nest{256,256};
    // Location de la nourriture
    position_t pos_food{500,500};
    //const int i_food = 500, j_food = 500;    
    // Génération du territoire 512 x 512 ( 2*(2^8) par direction )

    fractal_land land(8,2,1.,1024);
    double max_val = 0.0;
    double min_val = 0.0;
    for ( fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
        for ( fractal_land::dim_t j = 0; j < land.dimensions(); ++j ) {
            max_val = std::max(max_val, land(i,j));
            min_val = std::min(min_val, land(i,j));
        }
    double delta = max_val - min_val;
    /* On redimensionne les valeurs de fractal_land de sorte que les valeurs
    soient comprises entre zéro et un */
    for ( fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
        for ( fractal_land::dim_t j = 0; j < land.dimensions(); ++j )  {
            land(i,j) = (land(i,j)-min_val)/delta;
        }
    // Définition du coefficient d'exploration de toutes les fourmis.
    ant::set_exploration_coef(eps);
    // On va créer des fourmis un peu partout sur la carte :

    // Initialisation locale
    size_t ants_per_proc = nb_ants / size;
    if (rank == size - 1) ants_per_proc += nb_ants % size;

    std::vector<int> pos_x(ants_per_proc), pos_y(ants_per_proc);
    std::vector<char> loaded(ants_per_proc, false);
    std::vector<std::uint32_t> seeds(ants_per_proc);

    auto gen_ant_pos = [&land, &seed] () { return rand_int32(0, land.dimensions()-1, seed); };

    for ( size_t i = 0; i < ants_per_proc; ++i ){
        pos_x[i] = gen_ant_pos();
        pos_y[i] = gen_ant_pos();
        loaded[i] = 0;
        seeds[i] = seed + i;
    }
        
    // On crée toutes les fourmis dans la fourmilière.
    pheronome phen(land.dimensions(), pos_food, pos_nest, alpha, beta);

    // Calcular quantidades e displacements para Gatherv
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size, 0);

    for (int i = 0; i < size; ++i) {
        sendcounts[i] = (i < nb_ants % size) ? (nb_ants/size + 1) : (nb_ants/size);
        if (i > 0) displs[i] = displs[i-1] + sendcounts[i-1];
    }

    std::vector<int> all_pos_x(nb_ants);
    std::vector<int> all_pos_y(nb_ants);

    Window* win = nullptr;
    Renderer* renderer = nullptr;
    if (rank == 0) {
        win = new Window("Ant Simulation", 2*land.dimensions()+10, land.dimensions()+266);
        renderer = new Renderer(land, phen, pos_nest, pos_food, all_pos_x, all_pos_y);
    }

    // Compteur de la quantité de nourriture apportée au nid par les fourmis
    size_t local_food = 0;
    size_t global_food = 0;

    bool cont_loop = true;
    bool not_food_in_nest = true;

    std::size_t it = 0;

    double t_advance_ms = 0.0;
    double t_render_ms  = 0.0;

    while (cont_loop) {
        ++it;

        local_food = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        advance_time( land, phen, pos_nest, pos_food, pos_x, pos_y, loaded, seeds, local_food);
        auto t1 = std::chrono::high_resolution_clock::now();

        if (rank == 0) {
            MPI_Gatherv(pos_x.data(), pos_x.size(), MPI_INT, 
                    all_pos_x.data(), sendcounts.data(), displs.data(), MPI_INT,
                    0, MPI_COMM_WORLD);
            MPI_Gatherv(pos_y.data(), pos_y.size(), MPI_INT, 
                    all_pos_y.data(), sendcounts.data(), displs.data(), MPI_INT,
                    0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(pos_x.data(), pos_x.size(), MPI_INT, 
                    nullptr, nullptr, nullptr, MPI_INT,
                    0, MPI_COMM_WORLD);
            MPI_Gatherv(pos_y.data(), pos_y.size(), MPI_INT, 
                    nullptr, nullptr, nullptr, MPI_INT,
                    0, MPI_COMM_WORLD);
        }

        MPI_Allreduce(MPI_IN_PLACE, &global_food, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

        // Sincronizar feromônios entre processos
        phen.do_evaporation();
        phen.update();
        phen.sync_pheromones(MPI_COMM_WORLD);       

        auto t2 = std::chrono::high_resolution_clock::now();
        // Renderer rank 0
        if (rank == 0) {
            renderer->display(*win, global_food);   
            win->blit();
        }
        auto t3 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> d_advance = t1 - t0;
        std::chrono::duration<double> d_render = t3 - t2;

        t_advance_ms += d_advance.count() * 1000.0/n_iters;
        t_render_ms  += d_render .count() * 1000.0/n_iters;

        if ( not_food_in_nest && global_food > 0 ) {
            std::cout << "La première nourriture est arrivée au nid a l'iteration " << it << std::endl;
            not_food_in_nest = false;
        }
        if (it == n_iters) {
            cont_loop = false;
            std::cout << "Iter " << it << " advance_time = " << t_advance_ms << " ms, " << "render = " << t_render_ms << " ms" << std::endl;
        }
        //SDL_Delay(10);
    }
    if (rank == 0) SDL_Quit();
    MPI_Finalize();
    return 0;
}