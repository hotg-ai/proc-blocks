use ndarray::{ArrayD, ShapeError};

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

    pub fn view<T>(
        &self,
    ) -> Result<crate::ndarray::ArrayViewD<'_, T>, KernelError>
    where
        T: PrimitiveTensorElement,
    {
        if self.element_type != T::ELEMENT_TYPE {
            return Err(InvalidInput::unsupported_shape(&self.name).into());
        }

        let dimensions: Vec<_> =
            self.dimensions.iter().map(|&d| d as usize).collect();

        // Note: If our buffer is empty, the slice you get when from
        // the Deref implementation will be null + align_of(u8) with
        // a length of 0.
        //
        // This is normally fine, but if we later use bytemuck to
        // cast the &[u8] to &[T] and T has an alignment greater
        // than 1, we'll panic due to being mis-aligned.
        //
        // To prevent this, we return a view into an empty slice.
        if dimensions.iter().product::<usize>() == 0 {
            return crate::ndarray::ArrayViewD::from_shape(dimensions, &[])
                .map_err(|e| InvalidInput::other(&self.name, e).into());
        }

        let elements = bytemuck::try_cast_slice(&self.buffer)
                        .expect("Unable to reinterpret the buffer's bytes as the desired element type");

        crate::ndarray::ArrayViewD::from_shape(dimensions, elements)
            .map_err(|e| InvalidInput::other(&self.name, e).into())
    }

    pub fn view_with_dimensions<T, Dims>(
        &self,
    ) -> Result<crate::ndarray::ArrayView<'_, T, Dims>, KernelError>
    where
        T: PrimitiveTensorElement,
        Dims: crate::ndarray::Dimension,
    {
        self.view::<T>()?
            .into_dimensionality::<Dims>()
            .map_err(|e| InvalidInput::other(&self.name, e).into())
    }

    pub fn view_1d<T>(
        &self,
    ) -> Result<crate::ndarray::ArrayView1<'_, T>, KernelError>
    where
        T: PrimitiveTensorElement,
    {
        self.view_with_dimensions()
    }

    pub fn view_2d<T>(
        &self,
    ) -> Result<crate::ndarray::ArrayView2<'_, T>, KernelError>
    where
        T: PrimitiveTensorElement,
    {
        self.view_with_dimensions()
    }

    pub fn view_mut<T>(
        &mut self,
    ) -> Result<crate::ndarray::ArrayViewMutD<'_, T>, KernelError>
    where
        T: PrimitiveTensorElement,
    {
        if self.element_type != T::ELEMENT_TYPE {
            return Err(KernelError::InvalidInput(InvalidInput {
                name: self.name.clone(),
                reason: InvalidInputReason::UnsupportedShape,
            }));
        }

        let dimensions: Vec<_> =
            self.dimensions.iter().map(|&d| d as usize).collect();

        // See the comment in Tensor::view() for why we need this.
        if dimensions.iter().product::<usize>() == 0 {
            return crate::ndarray::ArrayViewMutD::from_shape(
                dimensions,
                &mut [],
            )
            .map_err(|e| InvalidInput::other(&self.name, e).into());
        }

        let elements = bytemuck::try_cast_slice_mut(&mut self.buffer)
                        .expect("Unable to reinterpret the buffer's bytes as the desired element type");

        crate::ndarray::ArrayViewMutD::from_shape(dimensions, elements)
            .map_err(|e| InvalidInput::other(&self.name, e).into())
    }

    pub fn view_with_dimensions_mut<T, Dims>(
        &mut self,
    ) -> Result<crate::ndarray::ArrayViewMut<'_, T, Dims>, KernelError>
    where
        T: PrimitiveTensorElement,
        Dims: crate::ndarray::Dimension,
    {
        // FIXME: It'd be nice if we didn't need to make this copy,
        // but the borrow checker isn't able to figure out that
        // the into_dimensionality() call consumes our view and
        // therefore the mutable borrow is finished.
        let name = self.name.clone();

        self.view_mut::<T>()?
            .into_dimensionality::<Dims>()
            .map_err(|e| InvalidInput::other(name, e).into())
    }

    pub fn string_view(&self) -> Result<ArrayD<&str>, ShapeError> {
        let dimensions: Vec<_> =
            self.dimensions.iter().map(|&dim| dim as usize).collect();
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
