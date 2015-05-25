use num::Zero;
use na::{Transform, Bounded};
use na;
use point::{LocalPointQuery, PointQuery};
use entities::bounding_volume::AABB;
use math::{Scalar, Point, Vect};

impl<P> LocalPointQuery<P> for AABB<P>
    where P: Point {
    #[inline]
    fn project_point(&self, pt: &P, solid: bool) -> P {
        let mins_pt = *self.mins() - *pt;
        let pt_maxs = *pt - *self.maxs();
        let shift   = na::sup(&na::zero(), &mins_pt) - na::sup(&na::zero(), &pt_maxs);

        if !shift.is_zero() || solid {
            *pt + shift
        }
        else {
            let _max: <P::Vect as Vect>::Scalar = Bounded::max_value();
            let mut best    = -_max;
            let mut best_id = 0isize;

            for i in 0 .. na::dim::<P::Vect>() {
                let mins_pt_i = mins_pt[i];
                let pt_maxs_i = pt_maxs[i];

                if mins_pt_i < pt_maxs_i {
                    if pt_maxs[i] > best {
                        best_id = i as isize;
                        best = pt_maxs_i
                    }
                }
                else if mins_pt_i > best {
                    best_id = -(i as isize);
                    best = mins_pt_i
                }
            }

            let mut shift: P::Vect = na::zero();

            if best_id < 0 {
                shift[(-best_id) as usize] = best;
            }
            else {
                shift[best_id as usize] = -best;
            }

            *pt + shift
        }
    }

    #[inline]
    fn distance_to_point(&self, pt: &P) -> <P::Vect as Vect>::Scalar {
        let mins_pt = *self.mins() - *pt;
        let pt_maxs = *pt - *self.maxs();

        na::norm(&na::sup(&na::zero(), &na::sup(&mins_pt, &pt_maxs)))
    }

    #[inline]
    fn contains_point(&self, pt: &P) -> bool {
        for i in 0 .. na::dim::<P>() {
            if pt[i] < self.mins()[i] || pt[i] > self.maxs()[i] {
                return false
            }
        }

        true
    }
}

impl<P, M> PointQuery<P, M> for AABB<P>
    where P: Point,
          M: Transform<P> {
}