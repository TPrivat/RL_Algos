import numpy as np


class Sum21:
    '''
    Sum21 Environment
    Keeps track of both dealers players hand
    '''
    def __init__(self):
        self.dealer_hand = [np.random.randint(1, 11)]
        self.player_hand = [np.random.randint(1, 11)]

    def get_card(self):
        num = np.random.randint(1, 11)
        color = np.random.random()

        if color > (1/3):
            return num              # Return a black card

        return -1 * num             # Return a red card

    def hit(self, hand):
        card = self.get_card()
        hand.append(card)

    def stay(self):
        # Ends players turn {TERMINAL STATE}
        # Dealer takes their turn
        # Reward is returned based on outcome of the game
        dealer_sum = self.dealers_turn()
        player_sum = self.get_sum(self.player_hand)

        if player_sum == 21:
            return 1

        if dealer_sum <= 0:
            return 1
        elif dealer_sum >= 22:
            return 1

        if dealer_sum > player_sum:
            return -1
        elif player_sum > dealer_sum:
            return 1
        else:
            return 0

    def get_sum(self, hand):
        rsum = 0
        for card in hand:
            rsum += card

        return rsum

    def dealers_turn(self):
        dsum = self.get_sum(self.dealer_hand)
        while dsum < 17 and dsum > 0:
            self.hit(self.dealer_hand)
            dsum = self.get_sum(self.dealer_hand)

        return dsum

    def get_state(self):
        # Returns the players hand sum and dealers seen card
        state = [self.get_sum(self.player_hand), self.dealer_hand[0]]

        return state

    def step(self, action):
        # Given action {0: hit, 1: stay}
        # returns True/False depending on if the game has (T) Terminated or (F) Continues
        # If hit will pull card and return reward depending on if player busted or not
        # If stay will have dealer go then return reward
        if action == 0:
            self.hit(self.player_hand)
            if self.get_sum(self.player_hand) > 21 or self.get_sum(self.player_hand) <= 0:
                return True, -1
            # if self.get_sum(self.player_hand) == 21:
            #     return True, 1
            return False, 0
        else:
            return True, self.stay()

    def reset(self):
        # Resets environment back to each player having one card
        self.dealer_hand = [np.random.randint(1, 11)]
        self.player_hand = [np.random.randint(1, 11)]

    def n_actions(self):
        return 2
