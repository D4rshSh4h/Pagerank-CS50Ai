import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probability = {page_name: 0 for page_name in corpus}
    links = len(corpus[page])
    if links == 0:
        # Handle case where there are no links in the corpus
        return {page: 1 / len(corpus) for page in corpus}
    # Calculate probabilities for each linked page
    #Probability as damping factor
    
    probability_damping = damping_factor/links

    #probability as uniform distribution
    probability_uniform = (1 - damping_factor) / len(corpus)

    #combine probabilities
    for page_name in probability:
        probability[page_name] += probability_uniform

        if page_name in corpus[page]:
            probability[page_name] += probability_damping

    return probability

def sample_pagerank(corpus, damping_factor, n):
    sample = dict(corpus)
    sample = {k: 0 for k in sample}
    #First sample
    page = random.choice(list(corpus.keys()))
    sample[page] += 1
    previous_model = transition_model(corpus, page, damping_factor)
    for x in range(0, n-1):
        next_page = str((random.choices(list(previous_model.keys()), weights = list(previous_model.values())))[0])
        sample[next_page] += 1
        previous_model = transition_model(corpus, next_page, damping_factor)
    
    
    ranks = {value: sample[value]/n for value in sample}
    return ranks



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    current_page_rank = {k: 1/len(corpus) for k in corpus}
    next_page_rank = {k: 0 for k in corpus}
    
    

    while True:
        max_diff = 0  
        for page in corpus:
            next_page_rank[page] = (1-damping_factor)/len(corpus.keys())
            
            for pages in corpus:
                if len(corpus[pages]) == 0:
                    next_page_rank[page] += damping_factor * current_page_rank[pages]/len(corpus.keys())
                    continue
                if page in corpus[pages]:
                    next_page_rank[page] +=  damping_factor * ((current_page_rank[pages])/len(corpus[pages]))
            
        norm_factor = sum(next_page_rank.values())
        next_page_rank = {page: rank/norm_factor for page, rank in next_page_rank.items()}
        max_diff = max(abs(next_page_rank[page] - current_page_rank[page]) for page in corpus)
        if max_diff < 0.001:
            break
        current_page_rank = dict(next_page_rank)
    return current_page_rank


if __name__ == "__main__":
    main()
