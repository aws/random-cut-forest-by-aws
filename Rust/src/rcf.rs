extern crate num;
extern crate rand;
extern crate rand_chacha;

use core::fmt::Debug;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_core::RngCore;
use rayon::prelude::*;

use crate::{
    common::{
        conditionalfieldsummarizer::FieldSummarizer, directionaldensity::InterpolationMeasure,
        divector::DiVector, samplesummary::SampleSummary,
    },
    l1distance,
    pointstore::{PointStore, VectorizedPointStore},
    samplerplustree::{
        nodestore::VectorNodeStore, nodeview::UpdatableNodeView, samplerplustree::SamplerPlusTree,
    },
    types::{Location, Result},
    util::{add_nbr, add_to, check_argument, divide, nbr_finish},
    visitor::{
        attributionvisitor::AttributionVisitor,
        imputevisitor::ImputeVisitor,
        interpolationvisitor::InterpolationVisitor,
        scalarscorevisitor::ScalarScoreVisitor,
        visitor::{Visitor, VisitorInfo},
    },
};
use crate::common::rangevector::RangeVector;

pub(crate) fn score_seen(x: usize, y: usize) -> f64 {
    1.0 / (x as f64 + f64::log2(1.0 + y as f64))
}
pub(crate) fn score_unseen(x: usize, _y: usize) -> f64 {
    1.0 / (x as f64 + 1.0)
}
pub(crate) fn normalizer(x: f64, y: usize) -> f64 {
    x * f64::log2(1.0 + y as f64)
}
pub(crate) fn damp(x: usize, y: usize) -> f64 {
    1.0 - (x as f64) / (2.0 * y as f64)
}

pub(crate) fn score_seen_displacement(_x: usize, y: usize) -> f64 {
    1.0 / (1.0 + y as f64)
}

// the following would be used for density estimation as well; note that for density estimation
// we are only focused about similarity of points and thus (previously) seen  and
// (previously) unseen points have little distinction
// that distinction can be crucial for some applications of anomaly detection

pub(crate) fn score_unseen_displacement(_x: usize, y: usize) -> f64 {
    y as f64
}

// the normalization is now multiplication by 1/treesize; this makes the
// max score to be 1.0 whereas for the standard score the average is close to 1

pub(crate) fn displacement_normalizer(x: f64, y: usize) -> f64 {
    x * 1.0 / (1.0 + y as f64)
}

pub(crate) fn identity(x: f64, _y: usize) -> f64 {
    x
}

pub trait RCF {
    fn update(&mut self, point: &[f32], timestamp: u64) -> Result<()>;

    fn dimensions(&self) -> usize;
    fn shingle_size(&self) -> usize;
    fn is_internal_shingling_enabled(&self) -> bool;
    fn entries_seen(&self) -> u64;
    fn size(&self) -> usize;
    fn point_store_size(&self) -> usize;
    fn shingled_point(&self,point:&[f32]) -> Vec<f32>;

    fn score(&self, point: &[f32]) -> Result<f64> {
        self.score_visitor_traversal(point, &VisitorInfo::default())
    }

    fn displacement_score(&self, point: &[f32]) -> Result<f64> {
        self.score_visitor_traversal(point, &VisitorInfo::displacement())
    }

    fn generic_score(
        &self,
        point: &[f32],
        ignore_mass: usize,
        score_seen: fn(usize, usize) -> f64,
        score_unseen: fn(usize, usize) -> f64,
        damp: fn(usize, usize) -> f64,
        normalizer: fn(f64, usize) -> f64,
    ) -> Result<f64> {
        self.score_visitor_traversal(
            point,
            &VisitorInfo::use_score(ignore_mass, score_seen, score_unseen, damp, normalizer),
        )
    }

    fn score_visitor_traversal(&self, point: &[f32], visitor_info: &VisitorInfo) -> Result<f64>;

    fn attribution(&self, point: &[f32]) -> Result<DiVector> {
        self.attribution_visitor_traversal(point, &VisitorInfo::default())
    }

    fn generic_attribution(
        &self,
        point: &[f32],
        ignore_mass: usize,
        score_seen: fn(usize, usize) -> f64,
        score_unseen: fn(usize, usize) -> f64,
        damp: fn(usize, usize) -> f64,
        normalizer: fn(f64, usize) -> f64,
    ) -> Result<DiVector> {
        self.attribution_visitor_traversal(
            point,
            &VisitorInfo::use_score(ignore_mass, score_seen, score_unseen, damp, normalizer),
        )
    }

    fn attribution_visitor_traversal(
        &self,
        point: &[f32],
        visitor_info: &VisitorInfo,
    ) -> Result<DiVector>;

    fn density(&self, point: &[f32]) -> Result<f64> {
        self.interpolation_visitor_traversal(point, &VisitorInfo::density())
            .map(|meas| meas.density())
    }

    fn directional_density(&self, point: &[f32]) -> Result<DiVector> {
        self.interpolation_visitor_traversal(point, &VisitorInfo::density())
            .map(|meas| meas.directional_density())
    }

    fn density_interpolant(&self, point: &[f32]) -> Result<InterpolationMeasure> {
        self.interpolation_visitor_traversal(point, &VisitorInfo::density())
    }

    fn interpolation_visitor_traversal(
        &self,
        point: &[f32],
        visitor_info: &VisitorInfo,
    ) -> Result<InterpolationMeasure>;

    /// the answer format is (score, point, distance from original)
    fn near_neighbor_list(
        &self,
        point: &[f32],
        percentile: usize,
    ) -> Result<Vec<(f64, Vec<f32>, f64)>> {
        self.near_neighbor_traversal(point, percentile, &VisitorInfo::default())
    }

    fn near_neighbor_traversal(
        &self,
        point: &[f32],
        percentile: usize,
        visitor_info: &VisitorInfo,
    ) -> Result<Vec<(f64, Vec<f32>, f64)>>;

    fn impute_missing_values(&self, positions: &[usize], point: &[f32]) -> Result<Vec<f32>> {
        assert!(positions.len() > 0, "nothing to impute");
        self.conditional_field(positions, point, 1.0, true, 0)
            .map(|summary| summary.median)
    }

    fn extrapolate(&self, look_ahead: usize) -> Result<RangeVector>;

    fn conditional_field(
        &self,
        positions: &[usize],
        point: &[f32],
        centrality: f64,
        project: bool,
        max_number: usize,
    ) -> Result<SampleSummary> {
        self.generic_conditional_field_visitor(
            positions,
            point,
            centrality,
            project,
            max_number,
            &VisitorInfo::default(),
        )
    }

    fn generic_conditional_field_visitor(
        &self,
        positions: &[usize],
        point: &[f32],
        centrality: f64,
        project: bool,
        max_number: usize,
        visitor_info: &VisitorInfo,
    ) -> Result<SampleSummary>;

    // to be extended to match Java version
}

pub struct RCFStruct<C, L, P, N>
where
    C: Location,
    usize: From<C>,
    L: Location,
    usize: From<L>,
    P: Location,
    usize: From<P>,
    N: Location,
    usize: From<N>,
{
    dimensions: usize,
    capacity: usize,
    number_of_trees: usize,
    sampler_plus_trees: Vec<SamplerPlusTree<C, P, N>>,
    time_decay: f64,
    shingle_size: usize,
    entries_seen: u64,
    internal_shingling: bool,
    internal_rotation: bool,
    store_attributes: bool,
    initial_accept_fraction: f64,
    bounding_box_cache_fraction: f64,
    parallel_enabled: bool,
    random_seed: u64,
    output_after: usize,
    point_store: VectorizedPointStore<L>,
}

pub type RCFTiny = RCFStruct<u8, u16, u16, u8>; // sampleSize <= 256 for these and shingleSize * { max { base_dimensions, (number_of_trees + 1) } <= 256
pub type RCFSmall = RCFStruct<u8, usize, u16, u8>; // sampleSize <= 256 and (number_of_trees + 1) <= 256 and dimensions = shingle_size*base_dimensions <= 256
pub type RCFMedium = RCFStruct<u16, usize, usize, u16>; // sampleSize, dimensions <= u16::MAX
pub type RCFLarge = RCFStruct<usize, usize, usize, usize>; // as large as the machine would allow

impl<C, L, P, N> RCFStruct<C, L, P, N>
where
    C: Location,
    usize: From<C>,
    L: Location,
    usize: From<L>,
    P: Location,
    usize: From<P>,
    N: Location,
    usize: From<N>,
    <C as TryFrom<usize>>::Error: Debug,
    <L as TryFrom<usize>>::Error: Debug,
    <P as TryFrom<usize>>::Error: Debug,
    <N as TryFrom<usize>>::Error: Debug,
{
    pub fn new(
        dimensions: usize,
        shingle_size: usize,
        capacity: usize,
        number_of_trees: usize,
        random_seed: u64,
        store_attributes: bool,
        parallel_enabled: bool,
        internal_shingling: bool,
        internal_rotation: bool,
        time_decay: f64,
        initial_accept_fraction: f64,
        bounding_box_cache_fraction: f64,
        output_after: usize,
    ) -> Self {
        let mut point_store_capacity: usize = (capacity * number_of_trees + 1).try_into().unwrap();
        if point_store_capacity < 2 * capacity {
            point_store_capacity = 2 * capacity;
        }
        let initial_capacity = 2 * capacity;
        if shingle_size != 1 && dimensions % shingle_size != 0 {
            println!("Shingle size must divide dimensions.");
            panic!();
        }
        assert!(
            !internal_rotation || internal_shingling,
            " internal shingling required for rotations"
        );
        let mut rng = ChaCha20Rng::seed_from_u64(random_seed);
        let _new_random_seed = rng.next_u64();
        let mut models: Vec<SamplerPlusTree<C, P, N>> = Vec::new();
        let using_transforms = internal_rotation; // other conditions may be added eventually
        for _i in 0..number_of_trees {
            models.push(SamplerPlusTree::<C, P, N>::new(
                dimensions,
                capacity,
                using_transforms,
                rng.next_u64(),
                store_attributes,
                time_decay,
                initial_accept_fraction,
                bounding_box_cache_fraction,
            ));
        }
        RCFStruct {
            random_seed,
            dimensions,
            capacity,
            sampler_plus_trees: models,
            number_of_trees,
            store_attributes,
            shingle_size,
            entries_seen: 0,
            time_decay,
            initial_accept_fraction,
            bounding_box_cache_fraction,
            parallel_enabled,
            point_store: VectorizedPointStore::<L>::new(
                dimensions.into(),
                shingle_size.into(),
                point_store_capacity,
                initial_capacity,
                internal_shingling,
                internal_rotation,
            ),
            internal_shingling,
            internal_rotation,
            output_after
        }
    }

    pub fn generic_conditional_field_point_list_and_distances(
        &self,
        positions: &[usize],
        point: &[f32],
        centrality: f64,
        visitor_info: &VisitorInfo,
    ) -> Vec<(f64, usize, f64)> {
        let new_point = self.point_store.get_shingled_point(point);
        let mut list: Vec<(f64, usize, f64)> = if self.parallel_enabled {
            self.sampler_plus_trees
                .par_iter()
                .map(|m| {
                    m.conditional_field(
                        &positions,
                        centrality,
                        &new_point,
                        &self.point_store,
                        visitor_info,
                    )
                })
                .collect()
        } else {
            self.sampler_plus_trees
                .iter()
                .map(|m| {
                    m.conditional_field(
                        &positions,
                        centrality,
                        &new_point,
                        &self.point_store,
                        visitor_info,
                    )
                })
                .collect()
        };
        list.sort_by(|&o1, &o2| o1.2.partial_cmp(&o2.2).unwrap());
        list
    }

    pub fn simple_traversal<NodeView, V, R, S>(
        &self,
        point: &[f32],
        parameters: &[usize],
        visitor_info: &VisitorInfo,
        visitor_factory: fn(usize, &[usize], &VisitorInfo) -> V,
        default: &R,
        initial: &S,
        collect_to: fn(&R, &mut S),
        finish: fn(&mut S, usize),
    ) -> Result<S>
    where
        NodeView: UpdatableNodeView<VectorNodeStore<C, P, N>, VectorizedPointStore<L>>,
        V: Visitor<NodeView, R>,
        R: Clone + std::marker::Send + std::marker::Sync,
        S: Clone,
    {
        check_argument(
            point.len() == self.dimensions || point.len() * self.shingle_size == self.dimensions,
            "invalid input length",
        )?;

        let mut answer = initial.clone();
        let new_point = self.point_store.get_shingled_point(point);

        if self.parallel_enabled {
            let list: Vec<R> = self
                .sampler_plus_trees
                .par_iter()
                .map(|m| {
                    //m.generic_visitor_traversal(
                    m.simple_traversal(
                        &new_point,
                        &self.point_store,
                        parameters,
                        &visitor_info,
                        visitor_factory,
                        default,
                    )
                })
                .collect();
            // given the overhead of parallelism, it seems appropriate to collect()
            // the below transformation is single threaded and the same function can be used
            // as is used in the single threaded case
            list.iter().for_each(|m| (collect_to)(m, &mut answer));
        } else {
            self.sampler_plus_trees
                .iter()
                .map(|m| {
                    m.simple_traversal(
                        &new_point,
                        &self.point_store,
                        parameters,
                        &visitor_info,
                        visitor_factory,
                        default,
                    )
                })
                .for_each(|m| collect_to(&m, &mut answer));
        }
        (finish)(&mut answer, self.sampler_plus_trees.len());
        Ok(answer)
    }
}

pub fn create_rcf(
    dimensions: usize,
    shingle_size: usize,
    capacity: usize,
    number_of_trees: usize,
    random_seed: u64,
    store_attributes: bool,
    parallel_enabled: bool,
    internal_shingling: bool,
    internal_rotation: bool,
    time_decay: f64,
    initial_accept_fraction: f64,
    bounding_box_cache_fraction: f64
) -> Box<dyn RCF> {
    if (dimensions < u8::MAX as usize) && (capacity - 1 <= u8::MAX as usize) {
        if capacity * (1 + number_of_trees) * shingle_size <= u16::MAX as usize {
            println!(" choosing RCF_Tiny");
            Box::new(RCFTiny::new(
                dimensions,
                shingle_size,
                capacity,
                number_of_trees,
                random_seed,
                store_attributes,
                parallel_enabled,
                internal_shingling,
                internal_rotation,
                time_decay,
                initial_accept_fraction,
                bounding_box_cache_fraction,
                capacity/4
            ))
        } else {
            println!(" choosing RCF_Small");
            Box::new(RCFSmall::new(
                dimensions,
                shingle_size,
                capacity,
                number_of_trees,
                random_seed,
                store_attributes,
                parallel_enabled,
                internal_shingling,
                internal_rotation,
                time_decay,
                initial_accept_fraction,
                bounding_box_cache_fraction,
                capacity/4
            ))
        }
    } else if (dimensions < u16::MAX as usize) && (capacity - 1 <= u16::MAX as usize) {
        println!(" choosing medium");
        Box::new(RCFMedium::new(
            dimensions,
            shingle_size,
            capacity,
            number_of_trees,
            random_seed,
            store_attributes,
            parallel_enabled,
            internal_shingling,
            internal_rotation,
            time_decay,
            initial_accept_fraction,
            bounding_box_cache_fraction,
            capacity/4
        ))
    } else {
        println!(" choosing large");
        Box::new(RCFLarge::new(
            dimensions,
            shingle_size,
            capacity,
            number_of_trees,
            random_seed,
            store_attributes,
            parallel_enabled,
            internal_shingling,
            internal_rotation,
            time_decay,
            initial_accept_fraction,
            bounding_box_cache_fraction,
            capacity/4
        ))
    }
}

impl<C, L, P, N> RCF for RCFStruct<C, L, P, N>
where
    C: Location,
    usize: From<C>,
    L: Location,
    usize: From<L>,
    P: Location,
    usize: From<P>,
    N: Location,
    usize: From<N>,
    <C as TryFrom<usize>>::Error: Debug,
    <L as TryFrom<usize>>::Error: Debug,
    <P as TryFrom<usize>>::Error: Debug,
    <N as TryFrom<usize>>::Error: Debug,
{
    fn shingled_point(&self,point:&[f32]) -> Vec<f32> {
        assert!(self.is_internal_shingling_enabled(), " incorrect function call");
        self.point_store.get_shingled_point(point)
    }

    fn update(&mut self, point: &[f32], _timestamp: u64) -> Result<()> {
        let point_index = self.point_store.add(&point);
        if point_index != usize::MAX {
            let result: Vec<(usize, usize)> = if self.parallel_enabled {
                self.sampler_plus_trees
                    .par_iter_mut()
                    .map(|m| m.update(point_index, usize::MAX, &self.point_store))
                    .collect()
            } else {
                self.sampler_plus_trees
                    .iter_mut()
                    .map(|m| m.update(point_index, usize::MAX, &self.point_store))
                    .collect()
            };
            self.point_store.adjust_count(&result);
            self.point_store.dec(point_index);
            self.entries_seen += 1;
        }
        Ok(())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn shingle_size(&self) -> usize {
        self.shingle_size
    }

    fn is_internal_shingling_enabled(&self) -> bool {
        self.internal_shingling
    }

    fn entries_seen(&self) -> u64 {
        self.entries_seen
    }

    fn score_visitor_traversal(&self, point: &[f32], visitor_info: &VisitorInfo) -> Result<f64> {
        // parameter unused for score traversal
        if self.output_after > self.entries_seen as usize {
            return Ok(0.0);
        }
        self.simple_traversal(
            point,
            &Vec::new(),
            visitor_info,
            ScalarScoreVisitor::Default,
            &0.0,
            &0.0,
            add_to,
            divide,
        )
    }

    fn attribution_visitor_traversal(
        &self,
        point: &[f32],
        visitor_info: &VisitorInfo,
    ) -> Result<DiVector> {
        if self.output_after > self.entries_seen as usize {
            return Ok(DiVector::empty(self.dimensions));
        }
        // tells the visitor what dimension to expect for each tree
        let parameters = &vec![self.dimensions];
        self.simple_traversal(
            point,
            parameters,
            visitor_info,
            AttributionVisitor::create_visitor,
            &DiVector::empty(self.dimensions),
            &DiVector::empty(self.dimensions),
            DiVector::add_to,
            DiVector::divide,
        )
    }

    fn interpolation_visitor_traversal(
        &self,
        point: &[f32],
        visitor_info: &VisitorInfo,
    ) -> Result<InterpolationMeasure> {
        // tells the visitor what dimension to expect for each tree
        let parameters = &vec![self.dimensions];
        self.simple_traversal(
            point,
            parameters,
            visitor_info,
            InterpolationVisitor::create_visitor,
            &InterpolationMeasure::empty(self.dimensions, 0.0),
            &InterpolationMeasure::empty(self.dimensions, 0.0),
            InterpolationMeasure::add_to,
            InterpolationMeasure::divide,
        )
    }

    fn near_neighbor_traversal(
        &self,
        point: &[f32],
        percentile: usize,
        visitor_info: &VisitorInfo,
    ) -> Result<Vec<(f64, Vec<f32>, f64)>> {
        let x = (0.0f64, usize::MAX, f64::MAX);
        let parameters = &vec![percentile];
        let list = self.simple_traversal(
            point,
            parameters,
            visitor_info,
            ImputeVisitor::create_nbr_visitor,
            &x,
            &Vec::new(),
            add_nbr,
            nbr_finish,
        )?;
        let mut answer = Vec::new();
        for e in list.iter() {
            answer.push((e.0, self.point_store.get_copy(e.1), e.2));
        }
        Ok(answer)
    }

    fn generic_conditional_field_visitor(
        &self,
        positions: &[usize],
        point: &[f32],
        centrality: f64,
        project: bool,
        max_number: usize,
        visitor_info: &VisitorInfo,
    ) -> Result<SampleSummary> {
        check_argument(
            point.len() == self.dimensions || point.len() * self.shingle_size == self.dimensions,
            "invalid input length",
        )?;
        let new_positions = if point.len() == self.dimensions {
            Vec::from(positions)
        } else {
            // internal shingling
            self.point_store.get_missing_indices(0, positions)
        };

        let raw_list = self.generic_conditional_field_point_list_and_distances(
            &new_positions,
            point,
            centrality,
            visitor_info,
        );
        let field_summarizer = FieldSummarizer::new(centrality, project, max_number, l1distance);
        Ok(field_summarizer.summarize_list(&self.point_store, &raw_list, &new_positions))
    }

    fn extrapolate(&self, look_ahead: usize) -> Result<RangeVector> {
        check_argument(
            self.internal_shingling,
            "look ahead is not meaningful without internal shingling mechanism",
        )?;
        check_argument(
            self.shingle_size > 1,
            "need shingle size > 1 for extrapolation",
        )?;
        let mut values = Vec::new();
        let mut upper = Vec::new();
        let mut lower = Vec::new();
        let base = self.dimensions / self.shingle_size;
        let mut fictitious_point = self.point_store.get_shingled_point(&vec![0.0f32; base]);
        for i in 0..look_ahead {
            let missing = self.point_store.get_next_indices(i);
            assert!(missing.len() == base, "incorrect imputation");
            let iterate = self.conditional_field(&missing, &fictitious_point, 1.0, true, 0).unwrap();
            for j in 0..base {
                values.push(iterate.median[j]);
                lower.push(iterate.lower[j]);
                upper.push(iterate.upper[j]);
                fictitious_point[missing[j]] = values[j];
            }
        }
        Ok(RangeVector::create(&values,&upper,&lower))
    }

    fn size(&self) -> usize {
        let mut sum: usize = 0;
        for model in &self.sampler_plus_trees {
            sum += model.get_size();
        }
        sum + self.point_store.get_size() + std::mem::size_of::<RCFStruct<C, L, P, N>>()
    }

    fn point_store_size(&self) -> usize {
        self.point_store.get_size()
    }
}
