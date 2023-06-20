import typing as t


def authors_abbr(authors: t.List[str]):
        ns = [(author.split()[0], author.split()[-1]) for author in authors]
        return [n[0]+'.'+s[0:2]+'.' for n, s in ns]