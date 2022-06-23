(function() {var implementors = {};
implementors["approx"] = [];
implementors["nalgebra"] = [{"text":"impl&lt;T, R:&nbsp;<a class=\"trait\" href=\"nalgebra/base/dimension/trait.Dim.html\" title=\"trait nalgebra::base::dimension::Dim\">Dim</a>, C:&nbsp;<a class=\"trait\" href=\"nalgebra/base/dimension/trait.Dim.html\" title=\"trait nalgebra::base::dimension::Dim\">Dim</a>, S&gt; <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>&lt;<a class=\"struct\" href=\"nalgebra/base/struct.Matrix.html\" title=\"struct nalgebra::base::Matrix\">Matrix</a>&lt;T, R, C, S&gt;&gt; for <a class=\"struct\" href=\"nalgebra/base/struct.Matrix.html\" title=\"struct nalgebra::base::Matrix\">Matrix</a>&lt;T, R, C, S&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: <a class=\"trait\" href=\"nalgebra/base/trait.Scalar.html\" title=\"trait nalgebra::base::Scalar\">Scalar</a> + <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;S: <a class=\"trait\" href=\"nalgebra/base/storage/trait.RawStorage.html\" title=\"trait nalgebra::base::storage::RawStorage\">RawStorage</a>&lt;T, R, C&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;T::<a class=\"associatedtype\" href=\"approx/abs_diff_eq/trait.AbsDiffEq.html#associatedtype.Epsilon\" title=\"type approx::abs_diff_eq::AbsDiffEq::Epsilon\">Epsilon</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,&nbsp;</span>","synthetic":false,"types":["nalgebra::base::matrix::Matrix"]},{"text":"impl&lt;T, R:&nbsp;<a class=\"trait\" href=\"nalgebra/base/dimension/trait.Dim.html\" title=\"trait nalgebra::base::dimension::Dim\">Dim</a>, C:&nbsp;<a class=\"trait\" href=\"nalgebra/base/dimension/trait.Dim.html\" title=\"trait nalgebra::base::dimension::Dim\">Dim</a>, S&gt; <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>&lt;<a class=\"struct\" href=\"nalgebra/base/struct.Unit.html\" title=\"struct nalgebra::base::Unit\">Unit</a>&lt;<a class=\"struct\" href=\"nalgebra/base/struct.Matrix.html\" title=\"struct nalgebra::base::Matrix\">Matrix</a>&lt;T, R, C, S&gt;&gt;&gt; for <a class=\"struct\" href=\"nalgebra/base/struct.Unit.html\" title=\"struct nalgebra::base::Unit\">Unit</a>&lt;<a class=\"struct\" href=\"nalgebra/base/struct.Matrix.html\" title=\"struct nalgebra::base::Matrix\">Matrix</a>&lt;T, R, C, S&gt;&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: <a class=\"trait\" href=\"nalgebra/base/trait.Scalar.html\" title=\"trait nalgebra::base::Scalar\">Scalar</a> + <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;S: <a class=\"trait\" href=\"nalgebra/base/storage/trait.RawStorage.html\" title=\"trait nalgebra::base::storage::RawStorage\">RawStorage</a>&lt;T, R, C&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;T::<a class=\"associatedtype\" href=\"approx/abs_diff_eq/trait.AbsDiffEq.html#associatedtype.Epsilon\" title=\"type approx::abs_diff_eq::AbsDiffEq::Epsilon\">Epsilon</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,&nbsp;</span>","synthetic":false,"types":["nalgebra::base::unit::Unit"]},{"text":"impl&lt;T:&nbsp;<a class=\"trait\" href=\"nalgebra/base/trait.Scalar.html\" title=\"trait nalgebra::base::Scalar\">Scalar</a> + <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>, D:&nbsp;<a class=\"trait\" href=\"nalgebra/base/dimension/trait.DimName.html\" title=\"trait nalgebra::base::dimension::DimName\">DimName</a>&gt; <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>&lt;<a class=\"struct\" href=\"nalgebra/geometry/struct.OPoint.html\" title=\"struct nalgebra::geometry::OPoint\">OPoint</a>&lt;T, D&gt;&gt; for <a class=\"struct\" href=\"nalgebra/geometry/struct.OPoint.html\" title=\"struct nalgebra::geometry::OPoint\">OPoint</a>&lt;T, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T::<a class=\"associatedtype\" href=\"approx/abs_diff_eq/trait.AbsDiffEq.html#associatedtype.Epsilon\" title=\"type approx::abs_diff_eq::AbsDiffEq::Epsilon\">Epsilon</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;<a class=\"struct\" href=\"nalgebra/base/default_allocator/struct.DefaultAllocator.html\" title=\"struct nalgebra::base::default_allocator::DefaultAllocator\">DefaultAllocator</a>: <a class=\"trait\" href=\"nalgebra/base/allocator/trait.Allocator.html\" title=\"trait nalgebra::base::allocator::Allocator\">Allocator</a>&lt;T, D&gt;,&nbsp;</span>","synthetic":false,"types":["nalgebra::geometry::point::OPoint"]},{"text":"impl&lt;T, const D:&nbsp;<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.usize.html\">usize</a>&gt; <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>&lt;<a class=\"struct\" href=\"nalgebra/geometry/struct.Rotation.html\" title=\"struct nalgebra::geometry::Rotation\">Rotation</a>&lt;T, D&gt;&gt; for <a class=\"struct\" href=\"nalgebra/geometry/struct.Rotation.html\" title=\"struct nalgebra::geometry::Rotation\">Rotation</a>&lt;T, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: <a class=\"trait\" href=\"nalgebra/base/trait.Scalar.html\" title=\"trait nalgebra::base::Scalar\">Scalar</a> + <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;T::<a class=\"associatedtype\" href=\"approx/abs_diff_eq/trait.AbsDiffEq.html#associatedtype.Epsilon\" title=\"type approx::abs_diff_eq::AbsDiffEq::Epsilon\">Epsilon</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,&nbsp;</span>","synthetic":false,"types":["nalgebra::geometry::rotation::Rotation"]},{"text":"impl&lt;T:&nbsp;<a class=\"trait\" href=\"nalgebra/trait.RealField.html\" title=\"trait nalgebra::RealField\">RealField</a> + <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>&lt;Epsilon = T&gt;&gt; <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>&lt;<a class=\"struct\" href=\"nalgebra/geometry/struct.Quaternion.html\" title=\"struct nalgebra::geometry::Quaternion\">Quaternion</a>&lt;T&gt;&gt; for <a class=\"struct\" href=\"nalgebra/geometry/struct.Quaternion.html\" title=\"struct nalgebra::geometry::Quaternion\">Quaternion</a>&lt;T&gt;","synthetic":false,"types":["nalgebra::geometry::quaternion::Quaternion"]},{"text":"impl&lt;T:&nbsp;<a class=\"trait\" href=\"nalgebra/trait.RealField.html\" title=\"trait nalgebra::RealField\">RealField</a> + <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>&lt;Epsilon = T&gt;&gt; <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>&lt;<a class=\"struct\" href=\"nalgebra/base/struct.Unit.html\" title=\"struct nalgebra::base::Unit\">Unit</a>&lt;<a class=\"struct\" href=\"nalgebra/geometry/struct.Quaternion.html\" title=\"struct nalgebra::geometry::Quaternion\">Quaternion</a>&lt;T&gt;&gt;&gt; for <a class=\"type\" href=\"nalgebra/geometry/type.UnitQuaternion.html\" title=\"type nalgebra::geometry::UnitQuaternion\">UnitQuaternion</a>&lt;T&gt;","synthetic":false,"types":["nalgebra::geometry::quaternion::UnitQuaternion"]},{"text":"impl&lt;T:&nbsp;<a class=\"trait\" href=\"nalgebra/trait.RealField.html\" title=\"trait nalgebra::RealField\">RealField</a> + <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>&lt;Epsilon = T&gt;&gt; <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>&lt;<a class=\"struct\" href=\"nalgebra/geometry/struct.DualQuaternion.html\" title=\"struct nalgebra::geometry::DualQuaternion\">DualQuaternion</a>&lt;T&gt;&gt; for <a class=\"struct\" href=\"nalgebra/geometry/struct.DualQuaternion.html\" title=\"struct nalgebra::geometry::DualQuaternion\">DualQuaternion</a>&lt;T&gt;","synthetic":false,"types":["nalgebra::geometry::dual_quaternion::DualQuaternion"]},{"text":"impl&lt;T:&nbsp;<a class=\"trait\" href=\"nalgebra/trait.RealField.html\" title=\"trait nalgebra::RealField\">RealField</a> + <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>&lt;Epsilon = T&gt;&gt; <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>&lt;<a class=\"struct\" href=\"nalgebra/base/struct.Unit.html\" title=\"struct nalgebra::base::Unit\">Unit</a>&lt;<a class=\"struct\" href=\"nalgebra/geometry/struct.DualQuaternion.html\" title=\"struct nalgebra::geometry::DualQuaternion\">DualQuaternion</a>&lt;T&gt;&gt;&gt; for <a class=\"type\" href=\"nalgebra/geometry/type.UnitDualQuaternion.html\" title=\"type nalgebra::geometry::UnitDualQuaternion\">UnitDualQuaternion</a>&lt;T&gt;","synthetic":false,"types":["nalgebra::geometry::dual_quaternion::UnitDualQuaternion"]},{"text":"impl&lt;T:&nbsp;<a class=\"trait\" href=\"nalgebra/trait.RealField.html\" title=\"trait nalgebra::RealField\">RealField</a>&gt; <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>&lt;<a class=\"struct\" href=\"nalgebra/base/struct.Unit.html\" title=\"struct nalgebra::base::Unit\">Unit</a>&lt;<a class=\"struct\" href=\"nalgebra/struct.Complex.html\" title=\"struct nalgebra::Complex\">Complex</a>&lt;T&gt;&gt;&gt; for <a class=\"type\" href=\"nalgebra/geometry/type.UnitComplex.html\" title=\"type nalgebra::geometry::UnitComplex\">UnitComplex</a>&lt;T&gt;","synthetic":false,"types":["nalgebra::geometry::unit_complex::UnitComplex"]},{"text":"impl&lt;T:&nbsp;<a class=\"trait\" href=\"nalgebra/base/trait.Scalar.html\" title=\"trait nalgebra::base::Scalar\">Scalar</a> + <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>, const D:&nbsp;<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.usize.html\">usize</a>&gt; <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>&lt;<a class=\"struct\" href=\"nalgebra/geometry/struct.Translation.html\" title=\"struct nalgebra::geometry::Translation\">Translation</a>&lt;T, D&gt;&gt; for <a class=\"struct\" href=\"nalgebra/geometry/struct.Translation.html\" title=\"struct nalgebra::geometry::Translation\">Translation</a>&lt;T, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T::<a class=\"associatedtype\" href=\"approx/abs_diff_eq/trait.AbsDiffEq.html#associatedtype.Epsilon\" title=\"type approx::abs_diff_eq::AbsDiffEq::Epsilon\">Epsilon</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,&nbsp;</span>","synthetic":false,"types":["nalgebra::geometry::translation::Translation"]},{"text":"impl&lt;T:&nbsp;<a class=\"trait\" href=\"nalgebra/trait.RealField.html\" title=\"trait nalgebra::RealField\">RealField</a>, R, const D:&nbsp;<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.usize.html\">usize</a>&gt; <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>&lt;<a class=\"struct\" href=\"nalgebra/geometry/struct.Isometry.html\" title=\"struct nalgebra::geometry::Isometry\">Isometry</a>&lt;T, R, D&gt;&gt; for <a class=\"struct\" href=\"nalgebra/geometry/struct.Isometry.html\" title=\"struct nalgebra::geometry::Isometry\">Isometry</a>&lt;T, R, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;R: <a class=\"trait\" href=\"nalgebra/geometry/trait.AbstractRotation.html\" title=\"trait nalgebra::geometry::AbstractRotation\">AbstractRotation</a>&lt;T, D&gt; + <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>&lt;Epsilon = T::<a class=\"associatedtype\" href=\"approx/abs_diff_eq/trait.AbsDiffEq.html#associatedtype.Epsilon\" title=\"type approx::abs_diff_eq::AbsDiffEq::Epsilon\">Epsilon</a>&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;T::<a class=\"associatedtype\" href=\"approx/abs_diff_eq/trait.AbsDiffEq.html#associatedtype.Epsilon\" title=\"type approx::abs_diff_eq::AbsDiffEq::Epsilon\">Epsilon</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,&nbsp;</span>","synthetic":false,"types":["nalgebra::geometry::isometry::Isometry"]},{"text":"impl&lt;T:&nbsp;<a class=\"trait\" href=\"nalgebra/trait.RealField.html\" title=\"trait nalgebra::RealField\">RealField</a>, R, const D:&nbsp;<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.usize.html\">usize</a>&gt; <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>&lt;<a class=\"struct\" href=\"nalgebra/geometry/struct.Similarity.html\" title=\"struct nalgebra::geometry::Similarity\">Similarity</a>&lt;T, R, D&gt;&gt; for <a class=\"struct\" href=\"nalgebra/geometry/struct.Similarity.html\" title=\"struct nalgebra::geometry::Similarity\">Similarity</a>&lt;T, R, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;R: <a class=\"trait\" href=\"nalgebra/geometry/trait.AbstractRotation.html\" title=\"trait nalgebra::geometry::AbstractRotation\">AbstractRotation</a>&lt;T, D&gt; + <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>&lt;Epsilon = T::<a class=\"associatedtype\" href=\"approx/abs_diff_eq/trait.AbsDiffEq.html#associatedtype.Epsilon\" title=\"type approx::abs_diff_eq::AbsDiffEq::Epsilon\">Epsilon</a>&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;T::<a class=\"associatedtype\" href=\"approx/abs_diff_eq/trait.AbsDiffEq.html#associatedtype.Epsilon\" title=\"type approx::abs_diff_eq::AbsDiffEq::Epsilon\">Epsilon</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,&nbsp;</span>","synthetic":false,"types":["nalgebra::geometry::similarity::Similarity"]},{"text":"impl&lt;T:&nbsp;<a class=\"trait\" href=\"nalgebra/trait.RealField.html\" title=\"trait nalgebra::RealField\">RealField</a>, C:&nbsp;<a class=\"trait\" href=\"nalgebra/geometry/trait.TCategory.html\" title=\"trait nalgebra::geometry::TCategory\">TCategory</a>, const D:&nbsp;<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.usize.html\">usize</a>&gt; <a class=\"trait\" href=\"approx/ulps_eq/trait.UlpsEq.html\" title=\"trait approx::ulps_eq::UlpsEq\">UlpsEq</a>&lt;<a class=\"struct\" href=\"nalgebra/geometry/struct.Transform.html\" title=\"struct nalgebra::geometry::Transform\">Transform</a>&lt;T, C, D&gt;&gt; for <a class=\"struct\" href=\"nalgebra/geometry/struct.Transform.html\" title=\"struct nalgebra::geometry::Transform\">Transform</a>&lt;T, C, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;<a class=\"struct\" href=\"nalgebra/base/dimension/struct.Const.html\" title=\"struct nalgebra::base::dimension::Const\">Const</a>&lt;D&gt;: <a class=\"trait\" href=\"nalgebra/base/dimension/trait.DimNameAdd.html\" title=\"trait nalgebra::base::dimension::DimNameAdd\">DimNameAdd</a>&lt;<a class=\"type\" href=\"nalgebra/base/dimension/type.U1.html\" title=\"type nalgebra::base::dimension::U1\">U1</a>&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;T::<a class=\"associatedtype\" href=\"approx/abs_diff_eq/trait.AbsDiffEq.html#associatedtype.Epsilon\" title=\"type approx::abs_diff_eq::AbsDiffEq::Epsilon\">Epsilon</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;<a class=\"struct\" href=\"nalgebra/base/default_allocator/struct.DefaultAllocator.html\" title=\"struct nalgebra::base::default_allocator::DefaultAllocator\">DefaultAllocator</a>: <a class=\"trait\" href=\"nalgebra/base/allocator/trait.Allocator.html\" title=\"trait nalgebra::base::allocator::Allocator\">Allocator</a>&lt;T, <a class=\"type\" href=\"nalgebra/base/dimension/type.DimNameSum.html\" title=\"type nalgebra::base::dimension::DimNameSum\">DimNameSum</a>&lt;<a class=\"struct\" href=\"nalgebra/base/dimension/struct.Const.html\" title=\"struct nalgebra::base::dimension::Const\">Const</a>&lt;D&gt;, <a class=\"type\" href=\"nalgebra/base/dimension/type.U1.html\" title=\"type nalgebra::base::dimension::U1\">U1</a>&gt;, <a class=\"type\" href=\"nalgebra/base/dimension/type.DimNameSum.html\" title=\"type nalgebra::base::dimension::DimNameSum\">DimNameSum</a>&lt;<a class=\"struct\" href=\"nalgebra/base/dimension/struct.Const.html\" title=\"struct nalgebra::base::dimension::Const\">Const</a>&lt;D&gt;, <a class=\"type\" href=\"nalgebra/base/dimension/type.U1.html\" title=\"type nalgebra::base::dimension::U1\">U1</a>&gt;&gt;,&nbsp;</span>","synthetic":false,"types":["nalgebra::geometry::transform::Transform"]}];
if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()