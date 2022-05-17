
def parse_config(path):
    """Parses the configuration file"""
    options = dict()
    options['gpus'] = '0'
    options['num_workers'] = '0'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    if 'dim' in options:
        dim = options['dim']
        dim = dim.split(',')
        options['dim'] = [int(d) for d in dim]
    return options

if __name__ == '__main__':
    tmp = parse_config('../config/test.cfg')

    import pdb;
    pdb.set_trace()