# Evolute
An evolutionary algorithm toolbox

## Documentation

**Evolute** is a simple tool for quick experimentation with evolutionary
algorithms for numerical optimization.

It defines a **population** of individuals, represented as floating point
vectors, and applies a configurable set of evolutionary **operators** to
them in a predefined order.

The order is the following:
1. **Selection**: a subset of individuals are discarded, depending on
their fitness value
2. **Reproduction**: the discarded individuals are replaced by new
individuals, somehow generated from the survivors
3. **Mutation**: mutate some of the individuals
4. **Update**: update the fitnesses of the individuals 

### Module: *population*

Different population types are defined here.
Every population type can accept the following constructor arguments:
- **loci**: number of elements in an individual's chromosome
- **fitness_wrapper**: instance of a class in evolute.evaluation
- **limit**: maximum number of individuals, defaults to 100
- **operators**: an instance of Operators, see defaults later
- **initializer**: instance of a class defined in evolute.initialization, optional

### Module: *evaluation*

Defines different wrappers for fitness functions.
**Fitness** is a scalar value assigned to every individual.
Currently in Evolute, the lower fitness is the better.

Since a fitness function can be any kind of logic, fitness wrappers
expect a function pointer or some kind of callable. Some advanced
wrappers are there to support possibly multiple fitness functions
or functions with multiple return values. In the end, the fitness
has to be reduced to a single scalar.

The process of combining multiple fitnesses to a single scalar is
termed **grading** in Evolute and so the more advanced fitness
wrappers are expecting a Grade function instance defined in
*evolute.evaluation.grade* or any callable which accepts a NumPy
array and returns a single scalar

Currently the following fitness wappers are defined in
*evolute.evaluation*:

- **SimpleFitness**: 