use coin_cbc::{raw::Status, Model, Sense};

fn main() {
    // a simple knapsack problem
    // Maximize  5x[1] + 3x[2] + 2x[3] + 7x[4] + 4x[5]
    // s.t.      2x[1] + 8x[2] + 4x[3] + 2x[4] + 5x[5] <= 10
    // All x binary

    // Create the problem.
    let mut m = Model::default();

    // The columns. m.add_binary() returns a typed indentifier for a
    // column.
    let cols = [
        m.add_binary(),
        m.add_binary(),
        m.add_binary(),
        m.add_binary(),
        m.add_binary(),
    ];

    // The row. m.add_row() returns a typed identifier for a row.
    let row = m.add_row();

    // Set the bound of the knapsack constraint.
    m.set_row_upper(row, 10.);

    // Set weights of the constraint.
    m.set_weight(row, cols[0], 2.);
    m.set_weight(row, cols[1], 8.);
    m.set_weight(row, cols[2], 4.);
    m.set_weight(row, cols[3], 2.);
    m.set_weight(row, cols[4], 5.);

    // Set objective coefficients
    m.set_obj_coeff(cols[0], 5.);
    m.set_obj_coeff(cols[1], 3.);
    m.set_obj_coeff(cols[2], 2.);
    m.set_obj_coeff(cols[3], 7.);
    m.set_obj_coeff(cols[4], 4.);

    // Set objective sense.
    m.set_obj_sense(Sense::Maximize);

    // Solve the problem. Returns the solution
    let sol = m.solve();

    // Check the result. sol.raw() returns a shared reference to the
    // raw bindings, allowing to use all getters.
    assert_eq!(Status::Finished, sol.raw().status());
    assert_eq!(16., sol.raw().obj_value());

    // Check for the solution.
    assert_eq!(1., sol.col(cols[0]));
    assert_eq!(0., sol.col(cols[1]));
    assert_eq!(0., sol.col(cols[2]));
    assert_eq!(1., sol.col(cols[3]));
    assert_eq!(1., sol.col(cols[4]));
}
