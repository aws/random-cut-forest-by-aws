extern crate num;
extern crate rand;
extern crate rand_chacha;

use core::fmt::Debug;
use std::collections::HashMap;
use std::hash::Hash;

use rand::{Rng, SeedableRng};
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
        nodeview::UpdatableNodeView, samplerplustree::SamplerPlusTree,
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
use crate::errors::RCFError;
use crate::errors::RCFError::InvalidArgument;

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

pub trait AugmentedRCF<Label,Attributes> {
    fn update(&mut self, point: &[f32], label: Label) -> Result<()>;
    fn id(&self) -> u64;
    fn dimensions(&self) -> usize;
    fn shingle_size(&self) -> usize;
    fn is_internal_shingling_enabled(&self) -> bool;
    fn is_output_ready(&self) -> bool;
    fn entries_seen(&self) -> u64;
    fn size(&self) -> usize;
    fn point_store_size(&self) -> usize;
    fn shingled_point(&self,point:&[f32]) -> Result<Vec<f32>>;

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
        self.interpolation_visitor_traversal(point, &VisitorInfo::density())?
            .density()
    }

    fn directional_density(&self, point: &[f32]) -> Result<DiVector> {
        self.interpolation_visitor_traversal(point, &VisitorInfo::density())?
            .directional_density()
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
        check_argument(positions.len() > 0, "nothing to impute")?;
        self.conditional_field(positions, point, 1.0, true, 0)
            .map(|summary| summary.median)
    }

    fn extrapolate(&self, look_ahead: usize) -> Result<RangeVector<f32>>;

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

}

//pub trait LabeledRCF<Update> : AugmentedRCF<Update,u64> {}
//impl<Update: Sync + Copy + Send, U> LabeledRCF<Update> for U where U : AugmentedRCF<Update,u64> {}


pub trait RCF : AugmentedRCF<u64,u64> + Send + Sync {}
impl<U> RCF for U where U : AugmentedRCF<u64,u64> + Send + Sync {}


pub struct RCFStruct<C, L, P, N, Label,Attributes>
where
    C: Location,
    usize: From<C>,
    L: Location,
    usize: From<L>,
    P: Location + Eq + Hash,
    usize: From<P>,
    N: Location,
    usize: From<N>,
    Label: Copy + Sync + Send,
    Attributes : Copy + Sync+ Hash + Eq + Send,
{
    id : u64,
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
    propagate_attributes: bool,
    initial_accept_fraction: f64,
    bounding_box_cache_fraction: f64,
    parallel_enabled: bool,
    random_seed: u64,
    output_after: usize,
    point_store: VectorizedPointStore<L,Label,Attributes>,
}

impl<C, L, P, N, Label, Attributes> RCFStruct<C, L, P, N, Label, Attributes>
where
    C: Location,
    usize: From<C>,
    L: Location,
    usize: From<L>,
    P: Location + Eq + Hash,
    usize: From<P>,
    N: Location,
    usize: From<N>,
    Label: Copy + Sync + Send,
    Attributes : Copy + Sync + Eq + Hash + Send,
    <C as TryFrom<usize>>::Error: Debug,
    <L as TryFrom<usize>>::Error: Debug,
    <P as TryFrom<usize>>::Error: Debug,
    <N as TryFrom<usize>>::Error: Debug,
{
    pub fn new(
        id : u64,
        dimensions: usize,
        shingle_size: usize,
        capacity: usize,
        number_of_trees: usize,
        random_seed: u64,
        store_attributes: bool,
        store_pointsum: bool,
        propagate_attributes : bool,
        parallel_enabled: bool,
        internal_shingling: bool,
        internal_rotation: bool,
        time_decay: f64,
        initial_accept_fraction: f64,
        bounding_box_cache_fraction: f64,
        output_after: usize,
        attribute_creator : fn(&[Label], Label) -> Result<Attributes>,
        attribute_to_vec : Option<fn(&Attributes) -> Result<Vec<f32>>>
    ) -> Result<Self> {
        let mut point_store_capacity= capacity * number_of_trees + 1;
        if point_store_capacity < 2 * capacity {
            point_store_capacity = 2 * capacity;
        }
        let initial_capacity = 2 * capacity;
        check_argument(shingle_size==1 || dimensions % shingle_size == 0, "Shingle size must divide dimensions")?;
        check_argument(!internal_rotation || internal_shingling,
            " internal shingling required for rotations")?;
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
                store_pointsum,
                propagate_attributes,
                time_decay,
                initial_accept_fraction,
                bounding_box_cache_fraction,
            )?);
        }
        Ok(RCFStruct {
            id,
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
            point_store: VectorizedPointStore::<L,Label,Attributes>::new(
                dimensions.into(),
                shingle_size.into(),
                point_store_capacity,
                initial_capacity,
                internal_shingling,
                internal_rotation,
                store_attributes,
                propagate_attributes,
                attribute_creator,
                attribute_to_vec
            )?,
            internal_shingling,
            internal_rotation,
            output_after,
            propagate_attributes
        })
    }

    pub fn generic_conditional_field_point_list_and_distances(
        &self,
        positions: &[usize],
        point: &[f32],
        centrality: f64,
        visitor_info: &VisitorInfo,
    ) -> Result<Vec<(f64, usize, f64)>> {
        let new_point = self.point_store.shingled_point(point)?;
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
                .collect::<Result<Vec<(f64, usize, f64)>>>()?
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
                .collect::<Result<Vec<(f64, usize, f64)>>>()?
        };
        list.sort_by(|&o1, &o2| o1.2.partial_cmp(&o2.2).expect("should be total order"));
        Ok(list)
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
        NodeView: UpdatableNodeView<Label,Attributes>,
        V: Visitor<NodeView, R>,
        R: Clone + Send + Sync,
        S: Clone,
    {
        check_argument(
            point.len() == self.dimensions || point.len() * self.shingle_size == self.dimensions,
            "invalid input length",
        )?;

        let mut answer = initial.clone();
        let new_point = self.point_store.shingled_point(point)?;

        let list: Vec<R> = if self.parallel_enabled {
             self.sampler_plus_trees
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
                .collect::<Result<Vec<R>>>()?
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
                }).collect::<Result<Vec<R>>>()?
        };
        list.iter().for_each(|m| (collect_to)(m, &mut answer));
        (finish)(&mut answer, self.sampler_plus_trees.len());
        Ok(answer)
    }
}

#[deprecated]
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
) -> Box<dyn RCF + Sync + Send> {
    RCFBuilder::<u64,u64>::new(dimensions/shingle_size,shingle_size)
        .tree_capacity(capacity)
        .number_of_trees(number_of_trees)
        .random_seed(random_seed)
        .store_attributes(store_attributes)
        .parallel_enabled(parallel_enabled)
        .internal_shingling(internal_shingling)
        .internal_rotation(internal_rotation)
        .time_decay(time_decay)
        .initial_accept_fraction(initial_accept_fraction)
        .bounding_box_cache_fraction(bounding_box_cache_fraction)
        .build_default().unwrap() //unwrap kept for deprecated function
}

impl<C, L, P, N,Label,Attributes> AugmentedRCF<Label,Attributes> for RCFStruct<C, L, P, N,Label, Attributes>
where
    C: Location,
    usize: From<C>,
    L: Location,
    usize: From<L>,
    P: Location + Eq + Hash,
    usize: From<P>,
    N: Location,
    usize: From<N>,
    Label: Copy + Sync + Send,
    Attributes : Copy + Sync+ Hash + Eq + Send,
    <C as TryFrom<usize>>::Error: Debug,
    <L as TryFrom<usize>>::Error: Debug,
    <P as TryFrom<usize>>::Error: Debug,
    <N as TryFrom<usize>>::Error: Debug,
{
    fn shingled_point(&self,point:&[f32]) -> Result<Vec<f32>> {
        self.point_store.shingled_point(point)
    }

    fn id(&self) -> u64 {
        self.id
    }

    fn update(&mut self, point: &[f32], label : Label) -> Result<()> {
        let (point_index,point_attribute,vector) = self.point_store.add(&point,label)?;
        if point_index != usize::MAX {
            let result: Vec<((usize, usize),(usize,usize))> = if self.parallel_enabled {
                self.sampler_plus_trees
                    .par_iter_mut()
                    .map(|m| m.update(point_index, point_attribute, &self.point_store))
                    .collect::<Result<Vec<((usize, usize), (usize, usize))>>>()?
            } else {
                    self.sampler_plus_trees
                        .iter_mut()
                        .map(|m| m.update(point_index, point_attribute, &self.point_store))
                        .collect::<Result<Vec<((usize,usize),(usize,usize))>>>()?
            };
            self.point_store.adjust_count(&result)?;
            self.point_store.dec(point_index,point_attribute)?;
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

    fn is_output_ready(&self) -> bool {
        ((self.output_after + if self.is_internal_shingling_enabled() {self.shingle_size - 1} else {0}) as u64) < self.entries_seen
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
            ScalarScoreVisitor::default,
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
            answer.push((e.0, self.point_store.copy(e.1)?, e.2));
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
            self.point_store.missing_indices(0, positions)?
        };

        let raw_list = self.generic_conditional_field_point_list_and_distances(
            &new_positions,
            point,
            centrality,
            visitor_info,
        )?;
        let field_summarizer = FieldSummarizer::new(centrality, project, max_number, l1distance);
        field_summarizer.summarize_list(&self.point_store, &raw_list, &new_positions)
    }

    fn extrapolate(&self, look_ahead: usize) -> Result<RangeVector<f32>> {
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
        let mut fictitious_point = self.point_store.shingled_point(&vec![0.0f32; base])?;
        for i in 0..look_ahead {
            let missing = self.point_store.next_indices(i)?;
            check_argument(missing.len() == base, "incorrect imputation")?;
            let iterate = self.conditional_field(&missing, &fictitious_point, 1.0, true, 0)?;
            for j in 0..base {
                values.push(iterate.median[j]);
                lower.push(iterate.lower[j]);
                upper.push(iterate.upper[j]);
                fictitious_point[missing[j]] = values[j];
            }
        }
        RangeVector::create(&values,&upper,&lower)
    }

    fn size(&self) -> usize {
        let mut sum: usize = 0;
        for model in &self.sampler_plus_trees {
            sum += model.get_size();
        }
        sum + self.point_store.size() + std::mem::size_of::<RCFStruct<C, L, P, N,Label,Attributes>>()
    }

    fn point_store_size(&self) -> usize {
        self.point_store.size()
    }
}



pub type RCFTiny<Update,Operate> = RCFStruct<u8, u16, u16, u8,Update,Operate>; // sampleSize <= 256 for these and shingleSize * { max { base_dimensions, (number_of_trees + 1) } <= 256
pub type RCFSmall<Update,Operate> = RCFStruct<u8, usize, u16, u8,Update,Operate>; // sampleSize <= 256 and (number_of_trees + 1) <= 256 and dimensions = shingle_size*base_dimensions <= 256
pub type RCFMedium<Update,Operate> = RCFStruct<u16, usize, usize, u16,Update,Operate>; // sampleSize, dimensions <= u16::MAX
pub type RCFLarge<Update,Operate> = RCFStruct<usize, usize, usize, usize,Update,Operate>; // as large as the machine would allow

pub fn copy_label_as_attribute<Label>(_x: &[Label],y:Label) -> Result<Label> {
    Ok(y)
}

pub struct RCFBuilder<Label : Send + Sync + Copy + 'static, Attributes : Send + Sync + Copy + Eq + Hash + 'static> {
    input_dimensions: usize,
    shingle_size: usize,
    pub(crate) rcf_options : RCFOptions<Label,Attributes>,
}


impl<Label : Send + Sync + Copy + 'static, Attributes : Send + Sync + Copy + Eq + Hash + 'static> RCFBuilder<Label,Attributes> {
    pub fn new(input_dimensions: usize, shingle_size: usize) -> Self {
        RCFBuilder {input_dimensions, shingle_size, rcf_options: RCFOptions::default()}
    }

    pub fn validate(&self) -> Result<()> {
        check_argument( self.input_dimensions > 0, "input_dimensions cannot be 0")?;
        check_argument( self.shingle_size > 0, "shingle size cannot be 0")?;
        self.rcf_options.validate()?;
        Ok(())
    }

    // coresion reasons
    pub fn build_default(&self) -> Result<Box<dyn RCF + Sync + Send>> {
        check_argument(self.rcf_options.attribute_to_vec.is_none(), "remove function options for default")?;
        check_argument(self.rcf_options.attribute_creator.is_none(), "remove function options fro default")?;

        let x =self.build_tiny_simple::<u64>();
        if x.is_ok() {
            Ok(Box::new(x?))
        } else {
            let y = self.build_small_simple::<u64>();
            if y.is_ok() {
                Ok(Box::new(y?))
            } else {
                let z = self.build_medium_simple::<u64>();
                if z.is_ok() {
                    Ok(Box::new(z?))
                } else {
                    Ok(Box::new(self.build_large_simple::<u64>()?))
                }
            }
        }
    }

    pub fn build_to_u64<Update: Send + Sync + Copy + 'static>(
        &self,
        attribute_creator : fn(&[Update],Update) -> Result<u64>
    ) -> Result<Box<dyn AugmentedRCF<Update,u64> + Sync + Send>> {
        let x =self.build_tiny::<Update,u64>(attribute_creator,None);
        if x.is_ok() {
            Ok(Box::new(x?))
        } else {
            let y = self.build_small::<Update,u64>(attribute_creator,None);
            if y.is_ok() {
                Ok(Box::new(y?))
            } else {
                let z = self.build_medium::<Update,u64>(attribute_creator, None);
                if z.is_ok() {
                    Ok(Box::new(z?))
                } else {
                    Ok(Box::new(self.build_large::<Update,u64>(attribute_creator,None)?))
                }
            }
        }
    }

    pub fn build(&self) -> Result<Box<dyn AugmentedRCF<Label,Attributes> + Sync + Send + 'static>> {
        check_argument(!self.rcf_options.store_attributes || self.rcf_options.attribute_creator.is_some(),
                       "need an attribute_creator function to create the attributes")?;
        let attribute_creator = self.rcf_options.attribute_creator.unwrap_or( |x,y |
            {Err(RCFError::InvalidArgument {msg : "function not provided, should not be invoked"})});
        let x =self.build_tiny::<Label,Attributes>(attribute_creator,self.rcf_options.attribute_to_vec);
        if x.is_ok() {
            Ok(Box::new(x?))
        } else {
            let y = self.build_small::<Label,Attributes>(attribute_creator,self.rcf_options.attribute_to_vec);
            if y.is_ok() {
                Ok(Box::new(y?))
            } else {
                let z = self.build_medium::<Label,Attributes>(attribute_creator, self.rcf_options.attribute_to_vec);
                if z.is_ok() {
                    Ok(Box::new(z?))
                } else {
                    Ok(Box::new(self.build_large::<Label,Attributes>(attribute_creator,self.rcf_options.attribute_to_vec)?))
                }
            }
        }
    }

    pub fn build_tiny<Update: Send + Sync + Copy, Operate:  Send + Sync + Copy + Eq + Hash>(
        &self,
        attribute_creator : fn(&[Update],Update) -> Result<Operate>,
        attribute_to_vec: Option<fn(&Operate) -> Result<Vec<f32>>>
    ) -> Result<RCFTiny<Update,Operate>> {
        self.validate()?;
        let dimensions = self.input_dimensions * self.shingle_size;
        let output_after = self.rcf_options.output_after.unwrap_or(1 + self.rcf_options.capacity / 4);
        let time_decay = self.rcf_options.time_decay.unwrap_or(0.1/self.rcf_options.capacity as f64);
        let random_seed = self.rcf_options.random_seed.unwrap_or( ChaCha20Rng::from_entropy().gen::<u64>());
        check_argument(dimensions < (u8::MAX as usize) && (self.rcf_options.capacity - 1 <= u8::MAX as usize),"incorrect parameters")?;
        check_argument(self.rcf_options.capacity * (1 + self.rcf_options.number_of_trees) * self.shingle_size <= u16::MAX as usize, " incorrect parameters")?;
        Ok(RCFTiny::<Update,Operate>::new(
            self.rcf_options.id,
            dimensions,
            self.shingle_size,
            self.rcf_options.capacity,
            self.rcf_options.number_of_trees,
            random_seed,
            self.rcf_options.store_attributes,
            self.rcf_options.store_pointsum,
            self.rcf_options.propagate_attributes,
            self.rcf_options.parallel_enabled,
            self.rcf_options.internal_shingling,
            self.rcf_options.internal_rotation,
            time_decay,
            self.rcf_options.initial_accept_fraction,
            self.rcf_options.bounding_box_cache_fraction,
            output_after,
            attribute_creator,
            attribute_to_vec
        )?)
    }

    pub fn build_tiny_simple<Operate: Send + Sync + Copy + Eq + Hash>(&self) -> Result<RCFTiny<Operate,Operate>> {
        self.build_tiny(copy_label_as_attribute::<Operate>,None)
    }

    pub fn build_small<Update: Send + Sync + Copy, Operate:  Send + Sync + Copy + Eq + Hash>(
        &self,
        attribute_creator : fn(&[Update],Update) -> Result<Operate>,
        attribute_to_vec: Option<fn(&Operate) -> Result<Vec<f32>>>
    ) -> Result<RCFSmall<Update,Operate>> {
        self.validate()?;
        let dimensions = self.input_dimensions * self.shingle_size;
        let output_after = self.rcf_options.output_after.unwrap_or(1 + self.rcf_options.capacity / 4);
        let time_decay = self.rcf_options.time_decay.unwrap_or(0.1 / self.rcf_options.capacity as f64);
        let random_seed = self.rcf_options.random_seed.unwrap_or(ChaCha20Rng::from_entropy().gen::<u64>());
        check_argument(dimensions < (u8::MAX as usize) && (self.rcf_options.capacity - 1 <= u8::MAX as usize), "incorrect parameters")?;
        Ok(RCFSmall::<Update,Operate>::new(
            self.rcf_options.id,
            dimensions,
            self.shingle_size,
            self.rcf_options.capacity,
            self.rcf_options.number_of_trees,
            random_seed,
            self.rcf_options.store_attributes,
            self.rcf_options.store_pointsum,
            self.rcf_options.propagate_attributes,
            self.rcf_options.parallel_enabled,
            self.rcf_options.internal_shingling,
            self.rcf_options.internal_rotation,
            time_decay,
            self.rcf_options.initial_accept_fraction,
            self.rcf_options.bounding_box_cache_fraction,
            output_after,
            attribute_creator,
            attribute_to_vec
        )?)
    }

    pub fn build_small_simple<Operate: Send + Sync + Copy + Eq + Hash>(&self) -> Result<RCFSmall<Operate,Operate>> {
        self.build_small(copy_label_as_attribute::<Operate>,None)
    }

    pub fn build_medium<Update: Send + Sync + Copy, Operate:  Send + Sync + Copy + Eq + Hash>(
        &self,
        attribute_creator : fn(&[Update],Update) -> Result<Operate>,
        attribute_to_vec: Option<fn(&Operate) -> Result<Vec<f32>>>
    ) -> Result<RCFMedium<Update,Operate>> {
        self.validate()?;
        let dimensions = self.input_dimensions * self.shingle_size;
        let output_after = self.rcf_options.output_after.unwrap_or(1 + self.rcf_options.capacity / 4);
        let time_decay = self.rcf_options.time_decay.unwrap_or(0.1 / self.rcf_options.capacity as f64);
        let random_seed = self.rcf_options.random_seed.unwrap_or(ChaCha20Rng::from_entropy().gen::<u64>());
        check_argument((dimensions < u16::MAX as usize) && (self.rcf_options.capacity - 1 <= u16::MAX as usize), " incorrect parameters")?;
        Ok(RCFMedium::<Update,Operate>::new(
            self.rcf_options.id,
            dimensions,
            self.shingle_size,
            self.rcf_options.capacity,
            self.rcf_options.number_of_trees,
            random_seed,
            self.rcf_options.store_attributes,
            self.rcf_options.store_pointsum,
            self.rcf_options.propagate_attributes,
            self.rcf_options.parallel_enabled,
            self.rcf_options.internal_shingling,
            self.rcf_options.internal_rotation,
            time_decay,
            self.rcf_options.initial_accept_fraction,
            self.rcf_options.bounding_box_cache_fraction,
            output_after,
            attribute_creator,
            attribute_to_vec
        )?)
    }

    pub fn build_medium_simple<Operate: Send + Sync + Copy + Eq + Hash>(&self) -> Result<RCFMedium<Operate,Operate>> {
        self.build_medium(copy_label_as_attribute::<Operate>,None)
    }

    pub fn build_large<Update: Send + Sync + Copy, Operate:  Send + Sync + Copy + Eq + Hash>(
        &self,
        attribute_creator : fn(&[Update],Update) -> Result<Operate>,
        attribute_to_vec: Option<fn(&Operate) -> Result<Vec<f32>>>
    ) -> Result<RCFLarge<Update,Operate>> {
        self.validate()?;
        let dimensions = self.input_dimensions * self.shingle_size;
        let output_after = self.rcf_options.output_after.unwrap_or(1 + self.rcf_options.capacity / 4);
        let time_decay = self.rcf_options.time_decay.unwrap_or(0.1/self.rcf_options.capacity as f64);
        let random_seed = self.rcf_options.random_seed.unwrap_or( ChaCha20Rng::from_entropy().gen::<u64>());
        Ok(RCFLarge::<Update,Operate>::new(
            self.rcf_options.id,
            dimensions,
            self.shingle_size,
            self.rcf_options.capacity,
            self.rcf_options.number_of_trees,
            random_seed,
            self.rcf_options.store_attributes,
            self.rcf_options.store_pointsum,
            self.rcf_options.propagate_attributes,
            self.rcf_options.parallel_enabled,
            self.rcf_options.internal_shingling,
            self.rcf_options.internal_rotation,
            time_decay,
            self.rcf_options.initial_accept_fraction,
            self.rcf_options.bounding_box_cache_fraction,
            output_after,
            attribute_creator,
            attribute_to_vec
        )?)
    }

    pub fn build_large_simple<Operate: Send + Sync + Copy + Eq + Hash>(&self) -> Result<RCFLarge<Operate,Operate>> {
        self.build_large(copy_label_as_attribute::<Operate>,None)
    }

}

pub struct RCFOptions<Label,Attributes> {
    pub(crate) id: u64,
    pub(crate) capacity: usize,
    pub(crate) number_of_trees: usize,
    pub(crate) time_decay: Option<f64>,
    pub(crate) internal_shingling: bool,
    pub(crate) internal_rotation: bool,
    pub(crate) store_labels: bool,
    pub(crate) store_attributes: bool,
    pub(crate) propagate_attributes: bool,
    pub(crate) store_pointsum: bool,
    pub(crate) initial_accept_fraction: f64,
    pub(crate) bounding_box_cache_fraction: f64,
    pub(crate) parallel_enabled: bool,
    pub(crate) random_seed: Option<u64>,
    pub(crate) output_after: Option<usize>,
    pub(crate) attribute_creator: Option<fn(&[Label],Label) -> Result<Attributes>>,
    pub(crate) attribute_to_vec: Option<fn(&Attributes) -> Result<Vec<f32>>>
}

impl<Label : Send + Sync + Copy, Attributes : Send + Sync + Copy + Eq + Hash> RCFOptions<Label,Attributes> {
    pub fn validate(&self) -> Result<()> {
        check_argument(self.capacity > 0, "capacity cannot be 0")?;
        check_argument( self.number_of_trees > 0, "number of trees cannot be 0")?;
        check_argument(self.time_decay.unwrap_or(0.0)>=0.0, "time decay cannot be negative")?;
        check_argument(self.bounding_box_cache_fraction >=0.0
                           && self.bounding_box_cache_fraction <=1.0,
                       "bounding box cache fraction is in [0,1]")?;
        check_argument(!self.propagate_attributes || self.store_attributes,
        "need to store attributes to propagate them")?;
        check_argument(self.initial_accept_fraction > 0.0 && self.initial_accept_fraction <= 1.0,
                       "initial accept fraction has to be in (0,1]")?;
        Ok(())
    }
}

impl<Label : Send + Sync + Copy, Attributes : Send + Sync + Copy + Eq + Hash> Default for RCFOptions<Label,Attributes> {
    fn default() -> Self {
        RCFOptions{
            id : u64::MAX, // a default tag that this was not set
            capacity: 256,
            number_of_trees: 50,
            time_decay: None,
            internal_shingling: true,
            internal_rotation: false,
            store_labels: false,
            store_attributes: false,
            propagate_attributes: false,
            initial_accept_fraction: 0.125,
            bounding_box_cache_fraction: 1.0,
            parallel_enabled: false,
            store_pointsum : false,
            random_seed: None,
            output_after: None,
            attribute_creator: Option::<fn( &[Label], Label) -> Result<Attributes>>::None,
            attribute_to_vec: Option::<fn( &Attributes) -> Result<Vec<f32>>>::None
        }
    }
}

pub trait RCFOptionsBuilder<Label : Send + Sync + Copy, Attributes : Send + Sync + Copy + Eq + Hash> {
    fn get_rcf_options(&mut self) -> &mut RCFOptions<Label,Attributes>;

    fn id(&mut self,id:u64) -> &mut Self{
        self.get_rcf_options().id = id;
        self
    }
    fn parallel_enabled(&mut self,parallel_enabled: bool) -> &mut Self {
        self.get_rcf_options().parallel_enabled = parallel_enabled;
        self
    }
    fn output_after(&mut self,output_after: usize) -> &mut Self {
        self.get_rcf_options().output_after = Some(output_after);
        self
    }
    fn random_seed(&mut self,random_seed: u64) -> &mut Self {
        self.get_rcf_options().random_seed = Some(random_seed);
        self
    }
    fn internal_rotation(&mut self, internal_rotation: bool) -> &mut Self {
        self.get_rcf_options().internal_rotation = internal_rotation;
        self
    }
    fn internal_shingling(&mut self, internal_shingling: bool) -> &mut Self {
        self.get_rcf_options().internal_shingling = internal_shingling;
        self
    }
    fn propagate_attribute_vectors(&mut self, propagarate_attribute_vectors : bool) -> &mut Self {
        self.get_rcf_options().propagate_attributes = propagarate_attribute_vectors;
        self
    }
    fn store_pointsum(&mut self, store_pointsum : bool) -> &mut Self {
        self.get_rcf_options().store_pointsum = store_pointsum;
        self
    }
    fn store_attributes(&mut self, store_attributes : bool) -> &mut Self {
        self.get_rcf_options().store_attributes = store_attributes;
        self
    }
    fn initial_accept_fraction(&mut self, initial_accept_fraction : f64) -> &mut Self {
        self.get_rcf_options().initial_accept_fraction = initial_accept_fraction;
        self
    }
    fn bounding_box_cache_fraction(&mut self, bounding_box_cache_fraction : f64) -> &mut Self {
        self.get_rcf_options().bounding_box_cache_fraction = bounding_box_cache_fraction;
        self
    }
    fn tree_capacity(&mut self, capacity: usize) -> &mut Self {
        self.get_rcf_options().capacity = capacity;
        self
    }

    fn number_of_trees(&mut self, number_of_trees : usize) -> &mut Self {
        self.get_rcf_options().number_of_trees = number_of_trees;
        self
    }
    fn time_decay(&mut self, time_decay : f64) -> &mut Self{
        self.get_rcf_options().time_decay = Some(time_decay);
        self
    }
    fn attribute_creator(&mut self, function: fn( &[Label],Label) -> Result<Attributes>) -> &mut Self{
        self.get_rcf_options().attribute_creator = Some(function);
        self
    }
    fn attribute_to_vec(&mut self, function: fn( _attribute :&Attributes) -> Result<Vec<f32>>) -> &mut Self{
        self.get_rcf_options().attribute_to_vec = Some(function);
        self
    }
}

impl<Label : Send + Sync + Copy, Attributes : Send + Sync + Copy + Eq + Hash> RCFOptionsBuilder<Label,Attributes> for RCFBuilder<Label,Attributes> {
    fn get_rcf_options(&mut self) -> &mut RCFOptions<Label,Attributes> {
        &mut self.rcf_options
    }
}
