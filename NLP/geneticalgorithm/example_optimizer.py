
import sys
from optimizer import GeneticAlgorithm

def color_back_red_index(x, index):
    """Formats the background of a string by using index instead of text"""
    import colorama
    list_x = list(x)
    for i_index in index:
        list_x[i_index] = colorama.Back.RED + list_x[i_index] + colorama.Style.NORMAL

    return ''.join(list_x)
	
	
	
# Start with two pieces of text that are almost identical
input_text_1 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore " \
               "et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut " \
               "aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse " \
               "cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."

input_text_2 = "Messing up the textLorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor " \
               "incididunt ut adding some more words labore et dolore magna aliqua. Ut enim ad minim veniam, " \
               "quis nostrud exercitation ullamco laboris nisi ut " \
               "aliquip ex ea commodo consequat. Duitsers Duis aute irure dolor in reprehenderit in voluptate velit " \
               "esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, " \
               "sunt in culpa see if it can find this as wellqui officia deserunt mollit anim id est laborum."

			   
# Set parameters of the model, and initalize the model

max_search = 50  # The max tolerance for a given character to find the other character
n_mut_repeat = 10  # How many time we mutate a child
mut_chance = 70  # The chance on mutation (in full percentages!!)
n_pop = 50  # The max amount of people in a generation/population
max_gen = 101  # The amount of times we generate new children... 

gen_object = GeneticAlgorithm(input_text_1, input_text_2, max_search=max_search, n_mut_repeat=n_mut_repeat, 
                     mut_chance=mut_chance,
                 n_pop=n_pop, max_gen=max_gen)
				 
# Capture the results

result = gen_object.run()  # This object contains the best scored individuals (distributions) 

# Score the best scoring individual in all generations
_, text_mapping = gen_object.score_individual(result[0])
x_index_list, y_index_list = zip(*text_mapping)

# Print the text mapping overlay...
print(color_back_red_index(gen_object.text_1, list(x_index_list)[1:]))
print('')
print(color_back_red_index(gen_object.text_2, list(y_index_list)[1:]))
