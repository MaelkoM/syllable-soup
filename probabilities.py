"""Generating names with a Markov p-matrix."""
import json
from random import randint
from tqdm import tqdm
import numpy as np

ORDER = 2
NAME_LENGTH = 8
NAME_COUNT = 2022
GENDER = "unisex"
COUNTRIES = ("us", "gb", "be", "fr", "es", "sa", "al", "bg", "de", "in", "other")


class MarkovGenerator:
    """Generating names with a Markov p-matrix."""

    def __init__(
        self,
        gender=GENDER,
        length=NAME_LENGTH,
        countries=COUNTRIES,
        random_length=False,
    ) -> None:
        self.order = ORDER
        self.length = length
        self.gender = gender
        self.names = None
        self.letters = None
        self.p_matrix = None
        self.profanities = None
        self.old_generated_names = None
        self.new_names = []
        self.countries = countries
        self.first_list = None
        self.prob_dict = {}
        self.name_dict = {}
        self.country_list = None
        self.random_length = random_length
        self.load_name_dict()
        self.load_country_list()
        self.choose_names_subset()
        self.create_letter_list()
        self.generate_first_list()
        self.get_old_generated_names()
        self.load_profanities()

    def load_country_list(self) -> None:
        """ Returns a list of countries saved in the specified .txt

        Returns:
            list: List of countries
        """
        self.country_list = list(
            set(self.name_dict[name]["country"] for name in self.name_dict)
        )

    def return_countries(self) -> list:
        """Returns a list of countries

        Returns:
            list: List of countries
        """
        return self.country_list

    def change_countries(self, countries: list) -> list:
        """ Changes the countries to be used in the generation

        Args:
            countries (list): List of countries

        Returns:
            list: List of countries
        """
        self.countries = countries

    def load_profanities(self) -> None:
        """ Loads the profanities from the specified .txt

        Returns:
            list: List of profanities
        """
        with open("data/profanity.json", "r") as prof_file:
            self.profanities = json.load(prof_file, encoding="utf-8")

    def change_order(self, order: int = 2) -> None:
        """ Changes the order of the Markov matrix

        Args:
            order (int): Order of the Markov matrix

        Returns:
            None
        """
        self.order = order

    def change_gender(self, gender: str = "unisex") -> None:
        """ Changes the gender of the names to be generated

        Args:
            gender (str): gender for the names
        """
        if gender != self.gender:
            self.gender = gender
            self.choose_names_subset()
            self.create_letter_list()

    def change_length(self, length: int = 8) -> None:
        """ Changes the length of the names to be generated

        Args:
            length (int): length of the names
        
        Returns:
            None
        """
        self.length = length

    def get_names(self) -> list:
        """ Returns a list of names

        Returns:
            list: List of names
        """
        return self.names

    def load_name_dict(self) -> list:
        """Returns a list of names saved in the specified .txt
        
        Returns:
            list: List of names
        """
        with open("data/names.json", "r") as name_file:
            self.name_dict = json.load(name_file)

    def choose_names_subset(self):
        """ Chooses a subset of names to be used in the generation

        Returns:
            None
        """
        if self.countries == []:
            self.countries = self.country_list
        if self.gender == "female":
            self.names = [
                name
                for name in self.name_dict
                if self.name_dict[name]["gender"] == "female"
                and len(name) > self.order
                and self.name_dict[name]["country"] in self.countries
            ]
        elif self.gender == "male":
            self.names = [
                name
                for name in self.name_dict
                if self.name_dict[name]["gender"] == "male"
                and len(name) > self.order
                and self.name_dict[name]["country"] in self.countries
            ]
        elif self.gender == "unisex":
            self.names = [
                name
                for name in self.name_dict
                if self.name_dict[name]["gender"] == "unisex"
                and len(name) > self.order
                and self.name_dict[name]["country"] in self.countries
            ]
        elif self.gender == "all":
            self.names = [
                name
                for name in self.name_dict
                if self.name_dict[name]["gender"] == "female"
                and len(name) > self.order
                and self.name_dict[name]["country"] in self.countries
            ]
            self.names.extend(
                [
                    name
                    for name in self.name_dict
                    if self.name_dict[name]["gender"] == "male"
                    and self.name_dict[name]["country"] in self.countries
                    and len(name) > self.order
                ]
            )
            self.names.extend(
                [
                    name
                    for name in self.name_dict
                    if self.name_dict[name]["gender"] == "unisex"
                    and self.name_dict[name]["country"] in self.countries
                    and len(name) > self.order
                ]
            )

    #        with open("data/echtenamen.txt") as name_file:
    #            self.names = [name.strip() for name in name_file.readlines()]

    def get_base_name_dict(self) -> dict:
        """ Returns a dictionary of names

        Returns:
            dict: Dictionary of names
        """
        return self.name_dict

    def generate_first_list(self) -> list:
        """ Generates a list of first names

        Returns:
            list: List of first names
        """
        self.first_list = [
            name[: self.order] for name in self.names if len(name) > self.order
        ]

    def get_old_generated_names(self) -> list:
        """ Returns a list of old generated names from the specified .txt

        Returns:
            list: List of old generated names
        """
        try:
            with open(f"data/neue_namen_{self.gender}.txt", mode="r") as name_file:
                self.old_generated_names = [
                    name.strip() for name in name_file.readlines()
                ]
        except:
            self.old_generated_names = []

    def create_letter_list(self) -> list:
        """ Creates a list of letters to be used in the generation of names from the names in the name list

        Returns:
            list: List of letters
        """
        if self.order == 1:
            letter_list = list(sorted(set("".join(set(self.names)))))
        else:
            letter_list = []
            for name in self.names:
                new_letters = [name[i : i + self.order] for i in range(0, len(name))]
                letter_list.extend(new_letters)
            letter_list = list(sorted(set(letter_list)))
        letter_list.append("stop")
        self.letters = letter_list

    def check_name(self, name: str) -> bool:
        """ Checks if the name is already in the list of names

        Args:
            name (str): Name to be checked

        Returns:
            bool: True if the name is already in the list of names
        """
        return bool(
            name not in self.names
            and name not in self.old_generated_names
            and name.lower() not in self.profanities
            and name not in self.new_names
        )

    def generate_new_name(self) -> list:
        """ Generates a new name from the names in the name list

        Returns:
            list: List of new names
        """
        state = np.random.choice(self.first_list)
        new_name = [state]
        new_name = self.accumulate_states(new_name, state)
        new_name = "".join(new_name)
        if self.check_name(name=new_name):
            self.new_names.append(new_name)
        else:
            self.generate_new_name()

    def accumulate_states(self, new_name: list, state: str) -> list:
        """ Accumulates states to generate a new name

        Args:
            new_name (list): list of states composing the new name
            state (str): state to be added to the new name

        Returns:
            list: Updated list of states composing the new name
        """
        while state != "stop":
            if state in self.prob_dict.keys():
                p_array = self.prob_dict[state]
            else:
                p_array = self.get_state_p_array(state)
                self.prob_dict[state] = p_array
            p_array = np.nan_to_num(p_array, 0.0001)
            p_array = p_array / np.sum(p_array)
            if len("".join(new_name)) + self.order <= self.length:
                try:
                    state = np.random.choice(self.letters, p=p_array)
                except ValueError:
                    break
                i = 0
                while (
                    len("".join(new_name)) < 3 / 4 * self.length
                    and (len(state) < self.order or state == "stop")
                    and i < 10
                ):
                    state = np.random.choice(self.letters, p=p_array)
                    i += 1
                if state != "stop":
                    new_name.append(state)
            else:
                state = np.random.choice(self.letters, p=p_array)
                i = 0
                while (
                    len(state) != self.length - len("".join(new_name))
                    and state != "stop"
                    and i < 10
                ):
                    state = np.random.choice(self.letters, p=p_array)
                    i += 1
                if state != "stop":
                    new_name.append(state)
                state = "stop"
            if new_name[-1] == "-":
                new_name = new_name[:-1]
        if self.random_length:
            self.length = randint(2, 14)
        return new_name

    def get_state_p_array(self, state: str) -> np.array:
        """ Returns a probability array for a state

        Args:
            state (str): State to be checked

        Returns:
            np.array: Probability array for the state
        """
        p_array = np.zeros(len(self.letters))
        for name in self.names:
            if state in name and len(name) > self.order:
                for index, syllable in enumerate(name):
                    if name[index : index + self.order] == state:
                        p_array = self.generate_p_array(
                            name_f=name, index_f=index, p_array_f=p_array
                        )
        return p_array

    def generate_p_array(self, name_f, index_f, p_array_f):
        """ Generates a probability array for a given state.

        Args:
            name_f (str): The name of the state.
            index_f (int): The index of the state in the name.
            p_array_f (np.array): The probability array of the state.

        Returns:
            np.array: The probability array of the state.
        """
        if name_f[index_f + self.order : index_f + self.order * 2]:
            new_index = np.asarray(
                np.array(self.letters)
                == name_f[index_f + self.order : index_f + self.order * 2]
            ).nonzero()
        else:
            new_index = np.asarray(np.array(self.letters) == "stop").nonzero()
        np.add.at(
            p_array_f, [new_index[0][0]], 1,
        )

        return p_array_f

    def return_new_names(self, n_names=NAME_COUNT) -> list:
        """ Returns a list of new names.

        Args:
            n_names (int): The number of names to return.

        Returns:
            list: A list of new names.
        """

        pbar = tqdm(total=n_names)
        i = 0
        while i < n_names:
            name_list_length = len(self.new_names)
            self.generate_new_name()
            pbar.update(len(self.new_names) - name_list_length)
            i += 1
        pbar.close()
        return self.new_names

    def save_names(
        self, names: list = None, json: bool = False, mode: str = "w+"
    ) -> None:
        """ Saves the names to a file.

        Args:
            names (list): The names to save.
            json (bool): Whether to save the names as json.
            mode (str): The mode to open the file in.

        Returns:
            None
        """

        if names is None:
            names = self.new_names
        print(f"Number of {self.gender} names:", len(names))
        if json == True:
            import json

            with open(f"data/neue_namen.json", mode=mode) as name_file:
                new_name_dict = {self.gender: names}
                json_obj = json.dumps(new_name_dict, ensure_ascii=False)
                name_file.write(json_obj)
        else:
            with open(f"data/neue_namen_{self.gender}.txt", mode=mode) as name_file:
                [name_file.write("%s\n" % name) for name in names]


class JaroChecker:
    """
    Test Jaro-Winkler distance metric.
    linuxwords.txt is from http://users.cs.duke.edu/~ola/ap/linuxwords
                    shuffle(list(set(names))).__dict__(), ensure_ascii=False
    """

    def jaro_winkler_distance(self, st1, st2):
        """
        Compute Jaro-Winkler distance between two strings.
        """
        if len(st1) < len(st2):
            st1, st2 = st2, st1
        len1, len2 = len(st1), len(st2)
        if len2 == 0:
            return 0.0
        delta = max(0, len2 // 2 - 1)
        flag = [False for _ in range(len2)]  # flags for possible transpositions
        ch1_match = []
        for idx1, ch1 in enumerate(st1):
            for idx2, ch2 in enumerate(st2):
                if (
                    idx1 - delta <= idx2 <= idx1 + delta
                    and ch1 == ch2
                    and not flag[idx2]
                ):
                    flag[idx2] = True
                    ch1_match.append(ch1)
                    break

        matches = len(ch1_match)
        if matches == 0:
            return 1.0
        transpositions, idx1 = 0, 0
        for idx2, ch2 in enumerate(st2):
            if flag[idx2]:
                transpositions += ch2 != ch1_match[idx1]
                idx1 += 1

        jaro = (
            matches / len1 + matches / len2 + (matches - transpositions / 2) / matches
        ) / 3.0
        commonprefix = 0
        for i in range(min(4, len2)):
            commonprefix += st1[i] == st2[i]

        return 1.0 - (jaro + commonprefix * 0.1 * (1 - jaro))

    def within_distance(self, maxdistance, name, maxtoreturn, corpus):
        """
        Find words in WORDS of closeness to name within maxdistance, return up to maxreturn of them.
        """
        arr = [
            word
            for word in corpus
            if self.jaro_winkler_distance(name, word) <= maxdistance
        ]
        arr.sort(key=lambda x: self.jaro_winkler_distance(name, x))
        return arr if len(arr) <= maxtoreturn else arr[:maxtoreturn]

    def check_new_names(self, new_names: list, corpus: list) -> list:
        """ Checks the new names for similarity to the corpus.

        Args:
            new_names (list): The new names to check.
            corpus (list): The corpus to check against.

        Returns:
            list: The new names that are similar to the corpus.
        """
        similarities = {}
        for name in new_names:
            for word in self.within_distance(0.9, name, 5, corpus):
                similarities[
                    word
                ] = f"{100 - round(self.jaro_winkler_distance(name, word)*100, 1)}%"
        return similarities


def get_countries() -> set:
    """ Returns a set of all countries.
    
        Returns:
            set: The set of all countries.
    """
    mark = MarkovGenerator()
    countries = list(
        set(
            [properties["country"] for properties in mark.get_base_name_dict().values()]
        )
    )
    with open("data/countries.txt", "w+") as country_file:
        [country_file.write(country + "\n") for country in countries]
    return countries


if __name__ == "__main__":
    genders = {
        "male": round(NAME_COUNT * 0.05),
        "unisex": round(NAME_COUNT * 0.05),
        "female": NAME_COUNT - (round(NAME_COUNT * 0.05) * 2),
    }
    for gender in tqdm(genders.keys()):
        print(f"Generating {genders[gender]} {gender} names!")
        countries = get_countries()
        countries.append(countries.remove("cn"))
        mark = MarkovGenerator(gender=gender, random_length=True, countries=countries,)
        generated_names = mark.return_new_names(genders[gender])
        mark.save_names(generated_names, json=False, mode="w")
        # jar = JaroChecker()[i][i]
        # jar.check_new_names(generated_names, mark.get_names())
