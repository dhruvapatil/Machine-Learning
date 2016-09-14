import os

'''
Input: Folder with file names of contents
Output: Sorted list with file_names
'''

def sort_files(folder_name = 'Frames'):
    '''Files in python are not sorted normally, they are sorted in the order in which the numbers appear:
    1. That means 1, 10, 100, 2, 20, 200...and so on...
    2. so we obtain the ending part of the filenames and then sort that array and return it.
    '''
    sorted_folder = sorted(map(int, [s.split('.')[0] for s in os.listdir(folder_name)]))
    return sorted_folder