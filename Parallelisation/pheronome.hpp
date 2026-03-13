#ifndef _PHERONOME_HPP_
#define _PHERONOME_HPP_
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>
#include "basic_types.hpp"
#include<mpi.h>

/**
 * @brief Carte des phéronomes
 * @details Gère une carte des phéronomes avec leurs mis à jour ( dont l'évaporation )
 *
 */
class pheronome {
public:
    using size_t      = unsigned long;
    using pheronome_t = std::array< double, 2 >;

    /**
     * @brief Construit une carte initiale des phéronomes
     * @details La carte des phéronomes est initialisées à zéro ( neutre )
     *          sauf pour les bords qui sont marqués comme indésirables
     *
     * @param dim Nombre de cellule dans chaque direction
     * @param alpha Paramètre de bruit
     * @param beta Paramêtre d'évaporation
     */
    pheronome( size_t dim, const position_t& pos_food, const position_t& pos_nest,
               double alpha = 0.7, double beta = 0.9999 )
        : m_dim( dim ),
          m_stride( dim + 2 ),
          m_alpha(alpha), m_beta(beta),
          m_map_of_pheronome( m_stride * m_stride, {{0., 0.}} ),
          m_buffer_pheronome( ),
          m_pos_nest( pos_nest ),
          m_pos_food( pos_food ) 
          {
        m_map_of_pheronome[index(pos_food)][0] = 1.;
        m_map_of_pheronome[index(pos_nest)][1] = 1.;
        cl_update( );
        m_buffer_pheronome = m_map_of_pheronome;
    }
    pheronome( const pheronome& ) = delete;
    pheronome( pheronome&& )      = delete;
    ~pheronome( )                 = default;

    pheronome_t& operator( )( size_t i, size_t j ) {
        return m_map_of_pheronome[( i + 1 ) * m_stride + ( j + 1 )];
    }

    const pheronome_t& operator( )( size_t i, size_t j ) const {
        return m_map_of_pheronome[( i + 1 ) * m_stride + ( j + 1 )];
    }

    pheronome_t& operator[] ( const position_t& pos ) {
      return m_map_of_pheronome[index(pos)];
    }

    const pheronome_t& operator[] ( const position_t& pos ) const {
      return m_map_of_pheronome[index(pos)];
    }

    void do_evaporation( ) {
        for ( std::size_t i = 1; i <= m_dim; ++i )
            for ( std::size_t j = 1; j <= m_dim; ++j ) {
                m_buffer_pheronome[i * m_stride + j][0] *= m_beta;
                m_buffer_pheronome[i * m_stride + j][1] *= m_beta;
            }
    }

    void mark_pheronome( const position_t& pos ) {
      std::size_t i = pos.x;
      std::size_t j = pos.y;
        assert( i >= 0 );
        assert( j >= 0 );
        assert( i < m_dim );
        assert( j < m_dim );
        pheronome&         phen        = *this;
        const pheronome_t& left_cell   = phen( i - 1, j );
        const pheronome_t& right_cell  = phen( i + 1, j );
        const pheronome_t& upper_cell  = phen( i, j - 1 );
        const pheronome_t& bottom_cell = phen( i, j + 1 );
        double             v1_left     = std::max( left_cell[0], 0. );
        double             v2_left     = std::max( left_cell[1], 0. );
        double             v1_right    = std::max( right_cell[0], 0. );
        double             v2_right    = std::max( right_cell[1], 0. );
        double             v1_upper    = std::max( upper_cell[0], 0. );
        double             v2_upper    = std::max( upper_cell[1], 0. );
        double             v1_bottom   = std::max( bottom_cell[0], 0. );
        double             v2_bottom   = std::max( bottom_cell[1], 0. );
        m_buffer_pheronome[( i + 1 ) * m_stride + ( j + 1 )][0] =
            m_alpha * std::max( {v1_left, v1_right, v1_upper, v1_bottom} ) +
            ( 1 - m_alpha ) * 0.25 * ( v1_left + v1_right + v1_upper + v1_bottom );
        m_buffer_pheronome[( i + 1 ) * m_stride + ( j + 1 )][1] =
            m_alpha * std::max( {v2_left, v2_right, v2_upper, v2_bottom} ) +
            ( 1 - m_alpha ) * 0.25 * ( v2_left + v2_right + v2_upper + v2_bottom );
    }

    void update( ) {
        m_map_of_pheronome.swap( m_buffer_pheronome );
        cl_update( );
        m_map_of_pheronome[( m_pos_food.x + 1 ) * m_stride + m_pos_food.y + 1][0] = 1;
        m_map_of_pheronome[( m_pos_nest.x + 1 ) * m_stride + m_pos_nest.y + 1][1] = 1;
    }
    
    void sync_pheromones(MPI_Comm comm) {
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        
        int total_size = m_stride * m_stride * 2;
        
        std::vector<double> all_pheromones;
        if (rank == 0) {
            all_pheromones.resize(total_size * size); 
        }
        
        std::vector<int> sendcounts(size, total_size);
        std::vector<int> displs(size, 0);
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i-1] + total_size;
        }
        
        // Envoie les données vers le processus 0
        MPI_Gatherv(m_map_of_pheronome.data(), total_size, MPI_DOUBLE,
                    rank == 0 ? all_pheromones.data() : nullptr,
                    sendcounts.data(), displs.data(), MPI_DOUBLE, 0, comm);
        
        // Calcul de la moyenne
        if (rank == 0) {
            for (auto& p : m_map_of_pheronome) {
                p = {0., 0.};
            }
            for (int proc = 0; proc < size; ++proc) {
                double* proc_data = &all_pheromones[proc * total_size];
                for (int i = 0; i < total_size/2; ++i) {
                    m_map_of_pheronome[i][0] += proc_data[2*i] / size;
                    m_map_of_pheronome[i][1] += proc_data[2*i + 1] / size;
                }
            }
            cl_update();
        }
        
        // Broadcast 
        MPI_Bcast(m_map_of_pheronome.data(), total_size, MPI_DOUBLE, 0, comm);
    }


private:
    size_t index( const position_t& pos ) const
    {
      return (pos.x+1)*m_stride + pos.y + 1;
    }
    /**
     * @brief Mets à jour les conditions limites sur les cellules fantômes
     * @details Mets à jour les conditions limites sur les cellules fantômes :
     *     pour l'instant, on se contente simplement de mettre ces cellules avec
     *     des valeurs à -1 pour être sûr que les fourmis évitent ces cellules
     */
    void cl_update( ) {
        // On mets tous les bords à -1 pour les marquer comme indésirables :
        for ( unsigned long j = 0; j < m_stride; ++j ) {
            m_map_of_pheronome[j]                            = {{-1., -1.}};
            m_map_of_pheronome[j + m_stride * ( m_dim + 1 )] = {{-1., -1.}};
            m_map_of_pheronome[j * m_stride]                 = {{-1., -1.}};
            m_map_of_pheronome[j * m_stride + m_dim + 1]     = {{-1., -1.}};
        }
    }
    unsigned long              m_dim, m_stride;
    double                     m_alpha, m_beta;
    std::vector< pheronome_t > m_map_of_pheronome, m_buffer_pheronome;
    position_t m_pos_nest, m_pos_food;
};

#endif