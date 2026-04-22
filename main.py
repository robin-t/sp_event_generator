try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    SIZE = 1

from density.density_handler import DensityHandler
from tune.tune_handler import TuneHandler
from tools.tickers import Tickers
from strategy.strategy_handler import StrategyHandler

import matplotlib as mpl

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["axes.unicode_minus"] = False


class MainApp:
    def __init__(self):
        self.tickers = Tickers()

        self.density_handler = DensityHandler(self.tickers)
        self.tune_handler = TuneHandler()
        self.strategy_handler = StrategyHandler(self.tune_handler)

    # --------------------------------------------------

    def show_menu(self):
        print("\n=== Stock Event Generator ===")
        print("1. Create density grids")
        print("2. Tune indicator weights")
        print("3. Test/run strategy")
        print("0. Exit")

    # --------------------------------------------------

    def run(self):
        while True:
            self.show_menu()
            choice = input("Select option: ")

            if choice == "1":
                self.density_handler.menu()

            elif choice == "2":
                self.tune_handler.menu()

            elif choice == "3":
                self.strategy_handler.menu()

            elif choice == "0":
                print("Goodbye 👋")
                break

            else:
                print("Invalid choice")


if __name__ == "__main__":

    app = MainApp()

    while True:

        # ----------------------------
        # Only rank 0 handles input
        # ----------------------------
        if RANK == 0:
            app.show_menu()
            choice = input("Select option: ")
        else:
            choice = None

        # ----------------------------
        # Broadcast choice to all ranks
        # ----------------------------
        if COMM is not None:
            choice = COMM.bcast(choice, root=0)

        # ----------------------------
        # Execute choice on ALL ranks
        # ----------------------------

        if choice == "1":
            app.density_handler.menu()

        elif choice == "2":
            app.tune_handler.menu()

        elif choice == "3":
            app.strategy_handler.menu()

        elif choice == "0":
            if RANK == 0:
                print("Exit")
            break

        else:
            if RANK == 0:
                print("Invalid choice")