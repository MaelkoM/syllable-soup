"""Generating names with a Markov p-matrix."""
import json
import numpy as np


ORDER = 2
NAME_LENGTH = 15
NAME_COUNT = 5
GENDER = "unisex"
COUNTRIES = ("us", "gb", "other")


class MarkovGenerator:
    """Generating names with a Markov p-matrix."""

    def __init__(self, gender=GENDER, length=NAME_LENGTH, countries=COUNTRIES) -> None:
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
        self.load_name_dict()
        self.load_country_list()
        self.choose_names_subset()
        self.create_letter_list()
        self.generate_first_list()
        self.get_old_generated_names()
        self.load_profanities()

    def load_country_list(self) -> list:
        """INSERT DOCSTRING."""
        self.country_list = list(
            set(self.name_dict[name]["country"] for name in self.name_dict)
        )

    def return_countries(self) -> list:
        """INSERT DOCSTRING."""
        return self.country_list

    def change_countries(self, countries: list) -> list:
        """INSERT DOCSTRING."""
        self.countries = countries

    def load_profanities(self) -> None:
        """INSERT DOCSTRING."""
        with open("data/profanity.json", "r") as prof_file:
            self.profanities = json.load(prof_file, encoding="utf-8")

    def change_order(self, order: int) -> None:
        """INSERT DOCSTRING."""
        self.order = order

    def change_gender(self, gender: str) -> None:
        """INSERT DOCSTRING."""
        if gender != self.gender:
            self.gender = gender
            self.choose_names_subset()
            self.create_letter_list()

    def change_length(self, length: int) -> None:
        """INSERT DOCSTRING."""
        self.length = length

    def get_names(self) -> list:
        """INSERT DOCSTRING."""
        return self.names

    def load_name_dict(self) -> list:
        """Returns a list of names saved in the specified .txt"""
        with open("data/names.json", "r") as name_file:
            self.name_dict = json.load(name_file)

    def choose_names_subset(self):
        """INSERT DOCSTRING."""
        if self.countries == []:
            self.countries = self.country_list
        print(self.countries)
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

    def generate_first_list(self) -> list:
        """INSERT DOCSTRING."""
        self.first_list = [
            name[: self.order] for name in self.names if len(name) > self.order
        ]

    def get_old_generated_names(self) -> list:
        """INSERT DOCSTRING."""
        with open("data/neue_namen.txt", mode="r+") as name_file:
            self.old_generated_names = [name.strip() for name in name_file.readlines()]

    def create_letter_list(self) -> list:
        """INSERT DOCSTRING."""
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
        """INSERT DOCSTRING."""
        return bool(
            name not in self.names
            and name not in self.old_generated_names
            and name not in self.profanities
        )

    def generate_new_name(self) -> list:
        """INSERT DOCSTRING."""
        state = np.random.choice(self.first_list)
        new_name = [state]
        new_name = self.accumulate_states(new_name, state)
        new_name = "".join(new_name)
        if self.check_name(name=new_name):
            self.new_names.append(new_name)

    def accumulate_states(self, new_name: list, state: str) -> list:
        """INSERT DOCSTRING."""
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
        return new_name

    def get_state_p_array(self, state: str) -> np.array:
        """INSERT DOCSTRING."""
        p_array = np.zeros(len(self.letters))
        for name in self.names:
            if state in name and len(name) > self.order:
                for index in enumerate(name):
                    if name[index : index + self.order] == state:
                        p_array = self.generate_p_array(
                            name_f=name, index_f=index, p_array_f=p_array
                        )
        return p_array

    def generate_p_array(self, name_f, index_f, p_array_f):
        """INSERT DOCSTRING."""
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
        """INSERT DOCSTRING."""
        i = 0
        while len(self.new_names) < n_names and i <= 100:
            self.generate_new_name()
            i += 1
            if i == 100:
                self.new_names = ["Let's generate a name!"]
        return self.new_names

    def save_names(self, names=None) -> None:
        """INSERT DOCSTRING."""
        if names is None:
            names = self.new_names
        with open("data/neue_namen.txt", mode="a") as name_file:
            for name in names:
                name_file.write("%s\n" % name)


class JaroChecker:
    """
    Test Jaro-Winkler distance metric.
    linuxwords.txt is from http://users.cs.duke.edu/~ola/ap/linuxwords
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

    def check_new_names(self, new_names, corpus):
        """INSERT DOCSTRING."""
        similarities = {}
        for name in new_names:
            for word in self.within_distance(0.9, name, 5, corpus):
                similarities[
                    word
                ] = f"{100 - round(self.jaro_winkler_distance(name, word)*100, 1)}%"
        return similarities


if __name__ == "__main__":
    mark = MarkovGenerator()
    generated_names = mark.return_new_names(2)
    mark.save_names(generated_names)
    jar = JaroChecker()
    jar.check_new_names(generated_names, mark.get_names())
