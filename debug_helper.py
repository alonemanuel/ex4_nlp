DECORATION_SYMBOL = '#'
N_LINE_BREAKS = 5
DECORATION_WIDTH = 20
LINE_BREAK = '\n'


def announce_section():
    print(LINE_BREAK * N_LINE_BREAKS)
    print(DECORATION_SYMBOL * DECORATION_WIDTH)


def announce_function(func_name, is_new_section=False):
    announce_section() if is_new_section else None
    print(f'In function: {func_name}')
    print(LINE_BREAK)
