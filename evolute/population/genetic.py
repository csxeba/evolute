from .population import Population


class GeneticPopulation(Population):

    def update_individual(self, index, **fitness_kw):
        self.fitnesses[index] = self.fitness(self.get_individual(index), **fitness_kw)
