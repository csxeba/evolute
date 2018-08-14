from .population import Population


class MemeticPopulation(Population):

    def update_individual(self, index, **fitness_kw):
        phenotype = self.get_individual(index)
        updated_individual, fitness = self.fitness(phenotype, **fitness_kw)
        self.fitnesses[index] = fitness
        self.individuals[index] = updated_individual
