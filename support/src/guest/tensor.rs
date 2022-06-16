use ndarray::{ArrayD, Dim, Dimension, IntoDimension, ShapeError};

use crate::{
    guest::{bindings::*, PrimitiveTensorElement},
    StringBuilder,
};

impl Tensor {
    pub fn new<T, S, Dims>(
        name: impl Into<String>,
        array: &crate::ndarray::ArrayBase<S, Dims>,
    ) -> Self
    where
        T: PrimitiveTensorElement,
        S: crate::ndarray::Data<Elem = T>,
        Dims: crate::ndarray::Dimension,
    {
        let dimensions = array.shape().iter().map(|&d| d as u32).collect();

        // Safety:
        let mut buffer = Vec::new();

        for element in array.iter() {
            buffer.extend(bytemuck::bytes_of(element));
        }

        Tensor {
            name: name.into(),
            dimensions,
            element_type: T::ELEMENT_TYPE,
            buffer,
        }
    }

    /// Serialize a string tensor so it can be passed to the runtime.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hotg_rune_proc_blocks::guest::Tensor;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let strings = ndarray::arr2(&[
    ///     ["this", "is", "a", "sentence"],
    ///     ["and", "this", "is", "another"],
    /// ]);
    ///
    /// let tensor = Tensor::from_strings("tensor", &strings);
    ///
    /// let deserialized = tensor.string_view()?;
    /// assert_eq!(deserialized, strings.into_dyn());
    /// # Ok(()) }
    /// ```
    pub fn from_strings<S, Data, Dim>(
        name: impl Into<String>,
        array: &ndarray::ArrayBase<Data, Dim>,
    ) -> Self
    where
        Dim: ndarray::Dimension,
        Data: ndarray::Data<Elem = S>,
        S: AsRef<str>,
    {
        let mut builder = StringBuilder::new();
        for s in array.iter() {
            builder.push(s.as_ref());
        }
        let buffer = builder.finish();

        let dimensions = array.shape().iter().map(|&dim| dim as u32).collect();

        Tensor {
            name: name.into(),
            element_type: ElementType::Utf8,
            dimensions,
            buffer,
        }
    }

    pub fn new_1d<T>(name: impl Into<String>, elements: &[T]) -> Self
    where
        T: PrimitiveTensorElement,
    {
        let array = crate::ndarray::aview1(elements);
        Tensor::new(name, &array)
    }

    pub fn with_name(self, name: impl Into<String>) -> Self {
        Tensor {
            name: name.into(),
            ..self
        }
    }

    pub fn take_named<'t>(
        tensors: &'t mut Vec<Tensor>,
        name: &str,
    ) -> Result<Self, RunError> {
        let index = tensors
            .iter()
            .position(|t| t.name == name)
            .ok_or_else(|| RunError::missing_input(name))?;

        Ok(tensors.remove(index))
    }

    pub fn get_named<'t>(
        tensors: &'t [Tensor],
        name: &str,
    ) -> Result<&'t Self, RunError> {
        tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| RunError::missing_input(name))
    }

    pub fn view<T>(
        &self,
    ) -> Result<crate::ndarray::ArrayViewD<'_, T>, InvalidInput>
    where
        T: PrimitiveTensorElement,
    {
        let dimensions: Vec<_> = self.dimensions().collect();
        let elements = self.elements()?;

        crate::ndarray::ArrayViewD::from_shape(dimensions, elements)
            .map_err(|e| InvalidInput::other(&self.name, e))
    }

    pub fn view_mut<T>(
        &mut self,
    ) -> Result<crate::ndarray::ArrayViewMutD<'_, T>, InvalidInput>
    where
        T: PrimitiveTensorElement,
    {
        let dimensions: Vec<_> = self.dimensions().collect();
        let name = self.name.clone();
        let elements = self.elements_mut()?;

        crate::ndarray::ArrayViewMutD::from_shape(dimensions, elements)
            .map_err(|e| InvalidInput::other(name, e))
    }

    pub fn view_with_dimensions_mut<T, const N: usize>(
        &mut self,
    ) -> Result<
        crate::ndarray::ArrayViewMut<'_, T, Dim<[usize; N]>>,
        InvalidInput,
    >
    where
        T: PrimitiveTensorElement,
        [usize; N]: IntoDimension<Dim = Dim<[usize; N]>>,
        Dim<[usize; N]>: Dimension,
    {
        let dimensions: [usize; N] = self.as_nd_shape()?;
        let name = self.name.clone();
        let elements = self.elements_mut()?;

        let shape = ndarray::Shape::from(ndarray::Dim(dimensions));

        crate::ndarray::ArrayViewMut::from_shape(shape, elements)
            .map_err(|e| InvalidInput::other(name, e))
    }

    pub fn view_with_dimensions<T, const N: usize>(
        &self,
    ) -> Result<crate::ndarray::ArrayView<'_, T, Dim<[usize; N]>>, InvalidInput>
    where
        T: PrimitiveTensorElement,
        [usize; N]: IntoDimension<Dim = Dim<[usize; N]>>,
        Dim<[usize; N]>: Dimension,
    {
        let dimensions: [usize; N] = self.as_nd_shape()?;
        let elements = self.elements()?;

        let shape = ndarray::Shape::from(ndarray::Dim(dimensions));

        crate::ndarray::ArrayView::from_shape(shape, elements)
            .map_err(|e| InvalidInput::other(&self.name, e))
    }

    fn elements<T>(&self) -> Result<&[T], InvalidInput>
    where
        T: PrimitiveTensorElement,
    {
        if self.element_type != T::ELEMENT_TYPE {
            return Err(
                InvalidInput::incompatible_element_type(&self.name).into()
            );
        }

        // Note: If our buffer is empty, the slice you get when from
        // the Deref implementation will be null + align_of(u8) with
        // a length of 0.
        //
        // This is normally fine, but if we later use bytemuck to
        // cast the &[u8] to &[T] and T has an alignment greater
        // than 1, we'll panic due to being mis-aligned.
        //
        // To prevent this, we return a view into an empty slice.

        if self.dimensions.iter().product::<u32>() == 0 {
            return Ok(&[]);
        }

        bytemuck::try_cast_slice(&self.buffer)
            .map_err(|e| InvalidInput::other(&self.name, e))
    }

    fn elements_mut<T>(&mut self) -> Result<&mut [T], InvalidInput>
    where
        T: PrimitiveTensorElement,
    {
        if self.element_type != T::ELEMENT_TYPE {
            return Err(
                InvalidInput::incompatible_element_type(&self.name).into()
            );
        }

        if self.dimensions.iter().product::<u32>() == 0 {
            return Ok(&mut []);
        }

        bytemuck::try_cast_slice_mut(&mut self.buffer)
            .map_err(|e| InvalidInput::other(&self.name, e))
    }

    fn dimensions(
        &self,
    ) -> impl Iterator<Item = usize> + DoubleEndedIterator + '_ {
        self.dimensions.iter().map(|&d| d as usize)
    }

    fn as_nd_shape<const N: usize>(&self) -> Result<[usize; N], InvalidInput> {
        let mut shape = [1; N];
        let mut last_index = N;

        for dim in self.dimensions().rev() {
            if dim == 1 {
                continue;
            }

            match last_index.checked_sub(1) {
                Some(ix) => last_index = ix,
                None => {
                    return Err(InvalidInput::incompatible_dimensions(
                        &self.name,
                    ));
                },
            }

            shape[last_index] = dim;
        }

        Ok(shape)
    }

    pub fn view_1d<T>(
        &self,
    ) -> Result<crate::ndarray::ArrayView1<'_, T>, InvalidInput>
    where
        T: PrimitiveTensorElement,
    {
        self.view_with_dimensions()
    }

    pub fn view_2d<T>(
        &self,
    ) -> Result<crate::ndarray::ArrayView2<'_, T>, InvalidInput>
    where
        T: PrimitiveTensorElement,
    {
        self.view_with_dimensions()
    }

    pub fn view_3d<T>(
        &self,
    ) -> Result<crate::ndarray::ArrayView3<'_, T>, InvalidInput>
    where
        T: PrimitiveTensorElement,
    {
        self.view_with_dimensions()
    }

    pub fn string_view(&self) -> Result<ArrayD<&str>, ShapeError> {
        let dimensions: Vec<_> = self.dimensions().collect();
        let strings = crate::strings::decode_strings(&self.buffer)?;

        ArrayD::from_shape_vec(dimensions, strings)
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Tensor) -> bool {
        let Tensor {
            name,
            element_type,
            dimensions,
            buffer,
        } = self;

        name == &other.name
            && element_type == &other.element_type
            && dimensions == &other.dimensions
            && buffer == &other.buffer
    }
}

impl From<Vec<u32>> for Dimensions {
    fn from(fixed: Vec<u32>) -> Self { Dimensions::Fixed(fixed) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewing_with_dimensionality_can_strip_or_add_leading_1s() {
        let elements = ndarray::arr2(&[[0.0_f64, 0.5, 10.0, 3.5, -200.0]]);
        let tensor = Tensor::new("x", &elements);

        // We should be able to view as both 1D, 2D, and 3D

        let view = tensor.view_1d::<f64>().unwrap();
        assert_eq!(view.dim(), 5);

        let view = tensor.view_2d::<f64>().unwrap();
        assert_eq!(view.dim(), (1, 5));

        let view = tensor.view_3d::<f64>().unwrap();
        assert_eq!(view.dim(), (1, 1, 5));
    }

    #[test]
    fn cant_view_2d_as_1d() {
        let elements = ndarray::arr2(&[[0.0_f64, 0.5], [10.0, -200.0]]);
        let tensor = Tensor::new("x", &elements);

        let err = tensor.view_1d::<f64>().unwrap_err();

        assert_eq!(err.reason, InvalidInputReason::IncompatibleDimensions);
    }
}
