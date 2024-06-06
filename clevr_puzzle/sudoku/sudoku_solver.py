# Code from https://github.com/ScriptRaccoon/sudoku-solver-python/blob/e672d8253792e7de04927f69c47786237363af1d/sudoku.py

"""Efficient Sudoku solver"""

from __future__ import annotations
from collections.abc import Iterator
from time import perf_counter


def key(row: int, col: int) -> str:
    """Encodes a coordinate such as (3,1) with the string 31"""
    return str(row) + str(col)


coords = {key(row, col) for row in range(9) for col in range(9)}
"""Set of all coordinates"""

row_units = [{key(row, col) for col in range(9)} for row in range(9)]
"""Lists of all rows as sets of coordinates"""

col_units = [{key(row, col) for row in range(9)} for col in range(9)]
"""List of all columns as sets of coordinates"""

box_units = [
    {key(3 * box_row + i, 3 * box_col + j) for i in range(3) for j in range(3)}
    for box_row in range(3)
    for box_col in range(3)
]
"""List of all boxes as sets of coordinates"""

all_units = row_units + col_units + box_units
"""List of all units (rows, columns, boxes)"""

peers: dict[str, set[str]] = {
    coord: set.union(*(unit - {coord} for unit in all_units if coord in unit))
    for coord in coords
}
"""Dictionary of all peers of a coordinate: the other coordinates that lie
in the same unit, and hence need to have different values in a Sudoku"""


class Sudoku:
    """Sudoku class"""

    def __init__(
        self,
        values: dict[str, int],
        candidates: dict[str, set[int]] | None = None,
    ) -> None:
        """Initialize a Sudoku with a value and candidate dictionaries

        Arguments:
            values: dictionary associating to each coordinate the digit at this square
            candidates: dictionary associating to each coordinate the set of its possible digits
        """
        self.values = values
        self.has_contradiction = False

        if candidates is None:
            self.candidates = self.get_candidate_dict()
        else:
            self.candidates = candidates

    @staticmethod
    def generate_from_board(
        board: list[list[int]],
    ) -> Sudoku:
        """Generates a Sudoku object from a given 2-dimensional list of integers"""
        values = {
            key(row, col): board[row][col] for row in range(9) for col in range(9)
        }
        return Sudoku(values)

    @staticmethod
    def generate_from_string(string: str) -> Sudoku:
        """Generates a Sudoku object from a one-line string as in the samples file"""
        string = string.replace("\n", "")
        assert len(string) == 81

        def to_digit(c: str) -> int:
            return int(c) if c.isnumeric() else 0

        values = {
            key(row, col): to_digit(string[row * 9 + col])
            for row in range(9)
            for col in range(9)
        }
        return Sudoku(values)

    def to_line(self) -> str:
        """Converts the Sudoku to a one-line string"""
        return "".join(map(str, list(self.values.values())))

    def __str__(self) -> str:
        """Computes a nice string representation of the Sudoku, used for printing to the console."""
        output = " " + "-" * 23 + "\n"
        for row in range(9):
            for col in range(9):
                digit = self.values[key(row, col)]
                if col == 0:
                    output += "| "
                output += (str(digit) if digit > 0 else ".") + " "
                if col % 3 == 2:
                    output += "| "
            output += "\n"
            if row % 3 == 2:
                output += " " + "-" * 23 + "\n"
        return output

    def copy(self) -> Sudoku:
        """Generates a copy of the Sudoku"""
        candidates_copy = {coord: self.candidates[coord].copy() for coord in coords}
        return Sudoku(self.values.copy(), candidates_copy)

    def get_candidates(self, coord: str) -> set[int]:
        """Generates the set of candidates at a coordinate"""
        digit = self.values[coord]
        if digit != 0:
            return {digit}
        values_of_peers = {self.values[peer] for peer in peers[coord]}
        return set(range(1, 10)) - values_of_peers

    def get_candidate_dict(self) -> dict[str, set[int]]:
        """Returns the dictionary of candidates for all coordinates"""
        return {coord: self.get_candidates(coord) for coord in coords}

    def get_next_coord(self) -> str | None:
        """Returns the free coordinate with the least number of candidates"""
        try:
            return min(
                (coord for coord in coords if self.values[coord] == 0),
                key=lambda coord: len(self.candidates[coord]),
            )
        except ValueError:
            return None

    def remove_candidate(self, coord: str, digit: int) -> None:
        """Removes a candidate from a coordinate (in case it's there),
        detects if a contradiction arises, and if a single candidate
        is left this cabdidate is set as a value."""
        if digit not in self.candidates[coord]:
            return
        self.candidates[coord].remove(digit)
        if not self.candidates[coord]:
            self.has_contradiction = True
        elif len(self.candidates[coord]) == 1:
            candidate = list(self.candidates[coord])[0]
            self.set_digit(coord, candidate)

    def set_digit(self, coord: str, digit: int) -> None:
        """Sets a digit at a given coordinate and removes that digit
        from the candidates of the coordinate's peers"""
        self.values[coord] = digit
        self.candidates[coord] = {digit}
        for peer in peers[coord]:
            self.remove_candidate(peer, digit)
            if self.has_contradiction:
                break

    def get_hidden_single(self) -> None | tuple[int, str]:
        """Returns a hidden single in a unit if present: a row, column or box
        where some digit has only one possible coordinate left"""
        for digit in range(1, 10):
            for unit in all_units:
                possible_coords = [
                    coord
                    for coord in unit
                    if self.values[coord] == 0 and digit in self.candidates[coord]
                ]
                if len(possible_coords) == 1:
                    return digit, possible_coords[0]
        return None

    def solutions(self) -> Iterator[Sudoku]:
        """Generates solutions of the given Sudoku"""

        # get and set hidden single
        single = self.get_hidden_single()
        if single:
            digit, coord = single
            self.set_digit(coord, digit)
            if not self.has_contradiction:
                yield from self.solutions()
            return

        # take coordinate with least number of candidates left
        next_coord = self.get_next_coord()
        if not next_coord:
            yield self
            return

        # test all candidates
        for candidate in self.candidates[next_coord]:
            copy = self.copy() if len(self.candidates[next_coord]) > 1 else self
            copy.set_digit(next_coord, candidate)
            if not copy.has_contradiction:
                yield from copy.solutions()


def measure_time() -> None:
    """Solves all sudoku samples and measures the time"""
    sudoku_counter: int = 0
    total: float = 0
    with open("data/solutions.txt", "w", encoding="utf8") as sol_file:
        with open("data/performance.txt", "w", encoding="utf8") as perf_file:
            with open("data/samples.txt", "r", encoding="utf8") as sample_file:
                for line in sample_file:
                    if line.startswith("#"):
                        continue
                    sudoku_counter += 1
                    sudoku = Sudoku.generate_from_string(line)
                    # print("solving sudoku", sudoku_counter)
                    start = perf_counter()
                    sols = list(sudoku.solutions())
                    end = perf_counter()
                    perf_file.write(str(end - start) + "\n")
                    total += end - start
                    assert len(sols) == 1
                    sol_file.write(sols[0].to_line() + "\n")
            perf_file.write("total: " + str(total) + "\n")
            average = total / sudoku_counter
            perf_file.write("average: " + str(average) + "\n")
    print("results written to data/performance.txt and data/solutions.txt")


def solve_sample(board):
    """Prints the solutions of a sample Sudoku"""
    sudoku = Sudoku.generate_from_board(board)
    # print("Sample:\n")
    # print(sudoku)
    # print("Solutions:\n")
    start = perf_counter()
    dict_solution = None
    for sol in sudoku.solutions():
        # print(sol)
        dict_solution = sol
    end = perf_counter()
    # print("Elapsed time: ", end - start, "\n")

    if dict_solution is None:
        # print("No solution found")
        return None, 0
    # convert solution into 2D array
    solution = [[0] * 9 for _ in range(9)]
    for coord, digit in dict_solution.values.items():
        row, col = map(int, coord)
        solution[row][col] = digit

    n_solutions = 0
    for _ in sudoku.solutions():
        n_solutions += 1

    return solution, n_solutions


if __name__ == "__main__":
    solve_sample()
    # measure_time()
