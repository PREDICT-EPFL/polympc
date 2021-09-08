// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef LINE_SEARCH_HPP
#define LINE_SEARCH_HPP

#include <list>
#include <iostream>

template<typename Scalar>
struct dominated_by
{
    dominated_by(const Scalar& _cost, const Scalar& _constr) : cost(_cost), constraint(_constr) {}

    Scalar cost{0};
    Scalar constraint{0};

    bool operator()(const std::pair<Scalar, Scalar>& candidate) const noexcept
    {
        return (candidate.first >= cost) && (candidate.second >= constraint);
    }
};

template<typename Scalar>
class LSFilter
{
public:
    using scalar_t = Scalar;
    using filter_pair_t = std::pair<Scalar, Scalar>;

    std::list<filter_pair_t> m_filter;
    int max_depth{10};
    scalar_t beta{1e-5};
    // debig print
    void print() const noexcept
    {
        std::cout << "Filter: \n";
        typename std::list<filter_pair_t>::const_iterator it;
        for(it = m_filter.begin(); it != m_filter.end(); ++it)
        {
            std::cout << "( " << it->first << " , " << it->second << " )" << "\n";
        }
    }

    //
    inline void clear() noexcept {m_filter.clear();}
    //
    inline bool is_dominated(const scalar_t& cost, const scalar_t& constraint) const noexcept
    {
        bool dominated = false;
        typename std::list<filter_pair_t>::const_iterator it;
        for(it = m_filter.begin(); it != m_filter.end(); ++it)
        {
            if((it->first <= cost) && (it->second <= constraint))
                return true;
        }
        return dominated;
    }
    //
    inline bool is_acceptable(const scalar_t& cost, const scalar_t& constraint) const noexcept
    {
        typename std::list<filter_pair_t>::const_iterator it;
        for(it = m_filter.begin(); it != m_filter.end(); ++it)
        {
            if (((it->first - beta * it->second)  <= cost) && ((it->second - beta * it->second) <= constraint))
                return false;
        }
        return true;
    }
    //
    inline void add(const scalar_t& cost, const scalar_t& constraint) noexcept
    {
        if(m_filter.size() < max_depth)
        {   // remove the points if the candidate point dominates them
            //m_filter.remove_if([cost, constraint](const filter_pair_t& candidate){return (candidate.first <= cost && candidate.second <= constraint);});
            m_filter.remove_if(dominated_by<scalar_t>(cost, constraint));
            m_filter.emplace_front(cost, constraint);
        }
        else
        {
            // clear the last element and push front
            //m_filter.sort();
            m_filter.pop_back();
            m_filter.emplace_front(cost, constraint);
        }
    }
    //
    inline void remove_one() noexcept
    {
        m_filter.pop_back();
    }
};




#endif // LINE_SEARCH_HPP
