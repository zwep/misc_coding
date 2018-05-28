# encoding: utf-8

"""
Copyright -  Seb Harrevelt (2018 - )
If you need help with this piece of code please contact me via sebharrevelt@gmail.com
"""

import random
from helper.miscfunction import diff_list


class GeneticAlgorithm:
    """
    Class to run an genetic algorithm against the comparison of two pieces of text. We are going to compare text_1
    with text_2. Such that every letter in text_1 is compared with text_2, a score is based on the amount of letters
    it gets correct.

    We could build this further such that we have one mother class defining the methods for the genetic algorithm,
    and one child-class that defines the scoring function and some other child-specfic things...

    Inspired on the work from
    https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9

    TODO: Add a measure for sparsity among the solution, so distributions with a high sparsity are discounted in
    their score. Need to find consecutive groups among the soluiton
    """

    def __init__(self, text_1, text_2, max_search, n_mut_repeat, mut_chance, n_pop, max_gen=100):
        self.text_1 = text_1
        self.text_2 = text_2
        self.length_text = len(self.text_1)
        self.n_mut_repeat = n_mut_repeat
        self.max_search = max_search
        self.max_gen = max_gen  # Stopping critera
        self.mut_chance = mut_chance  # Chance on mutation
        self.n_pop = n_pop  # Max population size
        self.n_best = int(0.05 * n_pop)
        self.n_worst = int(0.10 * n_pop)
        self.n_selection = self.n_best + self.n_worst
        self.init_population = self._gen_init_population()
        self.weight_tracker = []  # used to track weights over generations...

    def _gen_init_population(self):
        """ Here we create the initial population"""

        init_1 = [random.choice(range(self.max_search)) for x in range(self.length_text)]
        init_2 = [random.choice(range(self.max_search)) for x in range(self.length_text)]

        init_population = [init_1, init_2]
        while len(init_population) < self.n_pop:
            init_population.append(self.create_child(init_1, init_2))

        return init_population

    def _score_regulizer(self, current_pop):
        """ Piece of code to regularize the weights.. and decrease them. """
        total_weight = [sum(x) for x in current_pop]
        dev_weight = [self.length_text - x for x in total_weight]
        norm_weight = [0.05 * self.length_text * x/max(dev_weight) for x in dev_weight]
        return norm_weight

    def run(self):
        """ Runs the genetic algorithm """

        i_step = 1
        current_pop = self.init_population
        sorted_score = []
        sorted_id = []

        score_tracker = [0] * self.max_gen
        avg_diff_score = [999]
        n_hist = 4
        tol = 1

        reg_score = self._score_regulizer(current_pop)
        score_id = self.score_population(current_pop)
        reg_score_id = [(x[0] + reg_score[i], x[1]) for i, x in enumerate(score_id)]
        sorted_score, sorted_id = zip(*sorted(reg_score_id, reverse=True))

        while sum(avg_diff_score)/len(avg_diff_score) > tol and i_step < self.max_gen:
            if i_step % int(0.1*self.max_gen) == 0:
                print(i_step)

            reg_score = self._score_regulizer(current_pop)
            score_id = self.score_population(current_pop)
            reg_score_id = [(x[0] + reg_score[i], x[1]) for i, x in enumerate(score_id)]
            sorted_score, sorted_id = zip(*sorted(reg_score_id, reverse=True))

            sorted_population = [current_pop[i] for i in sorted_id]
            new_generation = self.select_from_population(sorted_population)

            n_child = 2*int((self.n_pop - self.n_selection)/self.n_selection)

            new_population = self.create_children(new_generation, n_child)
            current_pop = self.mutate_population(new_population)

            score_tracker[i_step] = sorted_score[0]

            avg_diff_score = diff_list(score_tracker[max(0, (i_step - n_hist)):(i_step+1)])
            avg_diff_score = [abs(x) for x in avg_diff_score]
            i_step += 1

        print('tolerance condition', sum(avg_diff_score)/len(avg_diff_score) > tol)
        print('max gen condition', i_step < self.max_gen)

        for i, j in zip(sorted_score[0:10], sorted_id[0:10]):
            print(i, '---', j)

        return current_pop

    def score_individual(self, current_distr):
        """
        Function to test some optimizer...
        :return:
        """
        import math

        score_mapping = 0
        text_mapping = [(-1, -1)]

        for x_origin, x_char in enumerate(self.text_1):
            start_point = int(text_mapping[-1][1]+1)
            end_point = int(math.ceil(start_point + current_distr[x_origin]))
            for y_origin, y_char in enumerate(self.text_2[start_point:end_point]):  # Give some area
                if x_char == y_char:
                    score_mapping += 1
                    text_mapping.append((x_origin, y_origin + text_mapping[-1][1] + 1))
                    break

        return score_mapping, text_mapping

    def score_population(self, population):
        """ Given a population of distributions, calculate the score """
        population_perf = []
        for i_id, individual in enumerate(population):
            population_perf.append((self.score_individual(individual)[0], i_id))
        return population_perf

    def select_from_population(self, population_sorted):
        """ Sample from the population to create children out of these """

        best_gen = [population_sorted[i] for i in range(self.n_best)]
        lucky_gen = [random.choice(population_sorted) for i in range(self.n_worst)]
        next_generation = best_gen + lucky_gen
        random.shuffle(next_generation)
        return next_generation

    @staticmethod
    def create_child(individual1, individual2):
        """ creating a child from two iterations """
        child = []
        for i in range(len(individual1)):
            if int(100 * random.random()) < 50:
                child.append(individual1[i])
            else:
                child.append(individual2[i])
        return child

    def create_children(self, breeders, n_child):
        """ Create a new population by 'breeding' a set of the population """
        next_population = []
        for i in range(int(len(breeders)/2)):
            for j in range(n_child):
                next_population.append(self.create_child(breeders[i], breeders[len(breeders) - 1 - i]))
        return next_population

    @staticmethod
    def mutate_individual(individual, n_max):
        """ Mutate the tolerance distribution"""
        index_modification = int(random.random() * len(individual))
        if index_modification == 0:
            individual = random.sample(range(1, n_max), 1) + individual[1:]
        else:
            individual = individual[:index_modification] + random.sample(range(1, n_max), 1) \
                         + individual[index_modification+1:]
        return individual

    def mutate_population(self, population):
        """ Mutate the whole population """
        for i in range(len(population)):
            if random.random() * 100 < self.mut_chance:
                for _ in range(self.n_mut_repeat):
                    population[i] = self.mutate_individual(population[i], self.max_search)
        return population
