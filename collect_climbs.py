import json

rows = reversed([str(i + 1) for i in range(18)])
cols = [chr(ord('A') + i) for i in range(11)]
coords = [c + r for r in rows for c in cols]

def get_holds(moves):
    matchstart = sum(1 for x in moves if x['isStart']) == 1

    starts = [x['description'] for x in moves if x['isStart']]
    middles = [x['description'] for x in moves if not x['isStart'] and not x['isEnd']]
    end = [x['description'] for x in moves if x['isEnd']]

    holds = [coords.index(x) for x in starts + middles + end]
    return matchstart, holds

def get_climb(climb):
    matchstart, holds = get_holds(climb['moves'])

    return {
        'name': climb['name'],
        'grade': climb['userGrade'] or climb['grade'],
        'setBy': climb['setby'],
        'method': climb['method'],
        'userRating': climb['userRating'],
        'repeats': climb['repeats'],
        'isBenchmark': climb['isBenchmark'],
        'matchStart': matchstart,
        'holds': holds
    }

def get_climbs(fn):
    with open(fn) as f:
        climbs = json.load(f)['data']
        print(f'{fn} has {len(climbs)} climbs')
        return [get_climb(climb) for climb in climbs]

def main():
    climbs = []
    for i in range(5):
        climbs += get_climbs(f'moon_files/moon{i}.json')

    print(f'exporting {len(climbs)} climbs')
    with open('data/climbs.json', 'w') as f:
        json.dump(climbs, f, indent = 4)

if __name__ == '__main__':
    main()