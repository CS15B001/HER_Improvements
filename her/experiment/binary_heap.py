#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 

import sys
import math

import utility


class BinaryHeap(object):

    def __init__(self, priority_size=100, priority_init=None, replace=True):
        # Experience_ID to Priority_ID (the node number in Heap) mapping
        self.e2p = {}
        # Priority_ID (the node number in Heap) to Experience_ID mapping
        self.p2e = {}
        # If the node should be replaced when inserting
        self.replace = replace

        if priority_init is None:
            self.priority_queue = {}
            self.size = 0
            self.max_size = priority_size
        else:
            # not yet tested
            self.priority_queue = priority_init
            self.size = len(self.priority_queue)
            self.max_size = None or self.size

            # map(func, list) returns a list after applying the function to all the elements 
            # The following is a list of (priority_IDs, experience_IDs)
            experience_list = list(map(lambda x: self.priority_queue[x], self.priority_queue))
            # Create the node_number --> Experience_ID and Experience_ID --> node_number mapping
            self.p2e = utility.list_to_dict(experience_list)
            self.e2p = utility.exchange_key_value(self.p2e)
            for i in range(int(self.size / 2), -1, -1):
                self.down_heap(i)

    def __repr__(self):
        """
        :return: string of the priority queue, with level info
        """

        # Gives a string representation of the heap for printing

        if self.size == 0:
            return 'No element in heap!'
        to_string = ''
        level = -1
        max_level = math.floor(math.log(self.size, 2))

        for i in range(1, self.size + 1):
            now_level = math.floor(math.log(i, 2))
            if level != now_level:
                to_string = to_string + ('\n' if level != -1 else '') \
                            + '    ' * int(max_level - now_level)
                level = now_level

            to_string = to_string + '%.2f ' % self.priority_queue[i][1] + '    ' * int(max_level - now_level)

        return to_string

    def check_full(self):
        # If the experience replay memory is full, return True
        return self.size > self.max_size

    def _insert(self, priority, e_id):
        """
        insert new experience id with priority
        (maybe don't need get_max_priority and implement it in this function)
        :param priority: priority value
        :param e_id: experience id
        :return: bool
        """
        self.size += 1
        
        if self.check_full() and not self.replace:
            sys.stderr.write('Error: no space left to add experience id %d with priority value %f\n' % (e_id, priority))
            # There should ideally be a self.size -= 1 here. Add it later
            return False
        else:
            # This statement will have an effect only if self.replace is true and the heap is full
            # Only then will the condition self.max_size == min(self.size, self.max_size) be true
            self.size = min(self.size, self.max_size)

        """
        Code will not do the intended when self.replace
        is True and size < max_size
        """

        # The first element of the node is the priority and the second its experience ID
        self.priority_queue[self.size] = (priority, e_id)
        self.p2e[self.size] = e_id
        # 1-based indexing
        self.e2p[e_id] = self.size

        # Restore the heap property
        self.up_heap(self.size)
        return True

    def update(self, priority, e_id):
        """
        update priority value according its experience id
        :param priority: new priority value
        :param e_id: experience id
        :return: bool
        """

        # Update the priority of a node and call up and down heap
        # to restore heap property
        if e_id in self.e2p:
            p_id = self.e2p[e_id]
            self.priority_queue[p_id] = (priority, e_id)
            self.p2e[p_id] = e_id

            self.down_heap(p_id)
            self.up_heap(p_id)
            return True
        else:
            # this e id is new, do insert
            return self._insert(priority, e_id)

    def get_max_priority(self):
        """
        get max priority, if no experience, return 1
        :return: max priority if size > 0 else 1
        """
        # Remember that the first element is ..[1] and not ..[0]
        # 1-based indexing

        if self.size > 0:
            return self.priority_queue[1][0]
        else:
            return 1

    def pop(self):
        """
        pop out the max priority value with its experience id
        :return: priority value & experience id
        """
        if self.size == 0:
            sys.stderr.write('Error: no value in heap, pop failed\n')
            return False, False

        # Remove the topmost node and replace it with the least priority
        # node. Then push the low priority node to its right position
        pop_priority, pop_e_id = self.priority_queue[1]
        self.e2p[pop_e_id] = -1
        # replace first
        last_priority, last_e_id = self.priority_queue[self.size]
        self.priority_queue[1] = (last_priority, last_e_id)
        self.size -= 1
        self.e2p[last_e_id] = 1
        self.p2e[1] = last_e_id

        self.down_heap(1)

        return pop_priority, pop_e_id

    def up_heap(self, i):
        """
        upward balance
        :param i: tree node i
        :return: None
        """

        # Move the child node up if its priority is greater than its parent
        # and continue recursively
        if i > 1:
            parent = math.floor(i / 2)
            if self.priority_queue[parent][0] < self.priority_queue[i][0]:
                tmp = self.priority_queue[i]
                self.priority_queue[i] = self.priority_queue[parent]
                self.priority_queue[parent] = tmp
                # change e2p & p2e

                # Make the necessary mapping changes because the 
                # child and parent nodes were swapped
                self.e2p[self.priority_queue[i][1]] = i
                self.e2p[self.priority_queue[parent][1]] = parent
                self.p2e[i] = self.priority_queue[i][1]
                self.p2e[parent] = self.priority_queue[parent][1]
                # up heap parent
                self.up_heap(parent)

    def down_heap(self, i):
        """
        downward balance
        :param i: tree node i
        :return: None
        """

        # Push the parent node down if its priority is lesser than
        # the child. Replace it with the greater of the two children
        if i < self.size:
            greatest = i
            left, right = i * 2, i * 2 + 1
            if left < self.size and self.priority_queue[left][0] > self.priority_queue[greatest][0]:
                greatest = left
            if right < self.size and self.priority_queue[right][0] > self.priority_queue[greatest][0]:
                greatest = right

            if greatest != i:
                tmp = self.priority_queue[i]
                self.priority_queue[i] = self.priority_queue[greatest]
                self.priority_queue[greatest] = tmp
                # change e2p & p2e
                self.e2p[self.priority_queue[i][1]] = i
                self.e2p[self.priority_queue[greatest][1]] = greatest
                self.p2e[i] = self.priority_queue[i][1]
                self.p2e[greatest] = self.priority_queue[greatest][1]
                # down heap greatest
                self.down_heap(greatest)

    def get_priority(self):
        """
        get all priority value
        :return: list of priority
        """

        # map(func, list) returns a list after applying the function to all the elements
        # This function thus returns all the priority values in the heap
        return list(map(lambda x: x[0], self.priority_queue.values()))[0:self.size]

    def get_e_id(self):
        """
        get all experience id in priority queue
        :return: list of experience ids order by their priority
        """

        # map(func, list) returns a list after applying the function to all the elements
        # This function thus returns all the experience ids in the heap
        return list(map(lambda x: x[1], self.priority_queue.values()))[0:self.size]

    def balance_tree(self):
        """
        rebalance priority queue
        :return: None
        """
        # Sort the array based on priority values
        sort_array = sorted(self.priority_queue.values(), key=lambda x: x[0], reverse=True)
        # reconstruct priority_queue, clear all variables first
        self.priority_queue.clear()
        self.p2e.clear()
        self.e2p.clear()
        cnt = 1
        while cnt <= self.size:
            priority, e_id = sort_array[cnt - 1]
            # Since a sorted array is a special case of priority queue just
            # add the elements to it one by one
            self.priority_queue[cnt] = (priority, e_id)
            # Create the node_number --> Experience_ID and Experience_ID --> node_number mapping
            self.p2e[cnt] = e_id
            self.e2p[e_id] = cnt
            cnt += 1
        # sort the heap

        # This seems unnecessary to me. The heap is already sorted
        # at the start of the function
        for i in range(int(math.floor(self.size / 2)), 1, -1):
            self.down_heap(i)

    def priority_to_experience(self, priority_ids):
        """
        retrieve experience ids by priority ids
        :param priority_ids: list of priority id
        :return: list of experience id
        """
        # The priority ids are just the node numbers
        # Retrieve the experience ids from the node numbers
        return [self.p2e[i] for i in priority_ids]
