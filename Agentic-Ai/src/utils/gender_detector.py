"""
gender_detector.py  –  Character Gender Detection

Priority order:
  1. Known female names dictionary  (RACHEL, SARAH, CLAIRE, etc.)
  2. Known male names dictionary    (JACK, JOHN, MIKE, etc.)
  3. Common female name endings     (-A, -IE, -INE, -ELLE, -ETH, etc.)
  4. Common male name endings       (-ER, -ON, -EY, -SON, etc.)
  5. Default → male (safest fallback for unknown names)
"""

from typing import Optional

# ---------------------------------------------------------------------------
# Known name dictionaries  (add more as your stories introduce new characters)
# ---------------------------------------------------------------------------

_KNOWN_FEMALE = {
    # A
    "ABIGAIL", "ADALINE", "ADRIANA", "AGATHA", "AGNES", "AILEEN", "AIMEE",
    "ALICE", "ALICIA", "ALINA", "ALISA", "ALISON", "ALISSA", "ALIYA",
    "ALLEGRA", "ALLISON", "ALMA", "AMANDA", "AMBER", "AMELIA", "AMY",
    "ANASTASIA", "ANDREA", "ANGELA", "ANGIE", "ANITA", "ANNA", "ANNE",
    "ANNETTE", "ANNIE", "ANTONIA", "APRIL", "ARABELLA", "ARIEL", "ARLENE",
    "ASHLEY", "ASTRID", "AUDREY", "AURORA", "AVA",
    # B
    "BARBARA", "BEATRICE", "BECKY", "BELLA", "BERNADETTE", "BETH",
    "BETTY", "BEVERLY", "BIANCA", "BONNIE", "BRENDA", "BRIDGET", "BRITTANY",
    # C
    "CAMILLA", "CANDICE", "CARLA", "CAROL", "CAROLINE", "CASSANDRA",
    "CASSIE", "CATHERINE", "CECILIA", "CELESTE", "CELINE", "CHARLOTTE",
    "CHERYL", "CHLOE", "CHRISTINA", "CHRISTINE", "CINDY", "CLAIRE",
    "CLARA", "CLAUDIA", "CLEMENTINE", "COLETTE", "CONSTANCE", "CORA",
    "COURTNEY", "CRYSTAL",
    # D
    "DAISY", "DANA", "DANIELA", "DANIELLE", "DAPHNE", "DEBORAH", "DEBRA",
    "DENISE", "DIANA", "DIANE", "DOMINIQUE", "DONNA", "DORIS", "DOROTHY",
    # E
    "EDEN", "EDITH", "ELAINE", "ELEANOR", "ELENA", "ELISA", "ELIZABETH",
    "ELLA", "ELLEN", "ELLIE", "ELSA", "EMILY", "EMMA", "ERICA", "ERIKA",
    "ERIN", "ESTELLE", "ESTHER", "EVA", "EVELYN", "EVIE",
    # F
    "FAITH", "FELICIA", "FELICITY", "FIONA", "FLORENCE", "FRANCES",
    "FRANCESCA", "FREYA",
    # G
    "GABRIELA", "GABRIELLE", "GEMMA", "GENEVIEVE", "GEORGIA", "GLORIA",
    "GRACE", "GRETA", "GWENDOLYN",
    # H
    "HANNAH", "HARRIET", "HAZEL", "HEATHER", "HELEN", "HELENA", "HILARY",
    "HOLLY", "HOPE",
    # I
    "INGRID", "IRENE", "IRIS", "ISABEL", "ISABELLA", "IVY",
    # J
    "JACQUELINE", "JADE", "JAMIE", "JANE", "JANET", "JASMINE", "JEAN",
    "JENNIFER", "JESSICA", "JILL", "JOAN", "JOANNA", "JOSEPHINE", "JOY",
    "JOYCE", "JULIA", "JULIET", "JUNE",
    # K
    "KAREN", "KATE", "KATHERINE", "KATHLEEN", "KATHRYN", "KATIE", "KATYA",
    "KIM", "KIRA", "KRISTEN", "KRISTINA",
    # L
    "LAURA", "LAUREN", "LEAH", "LEILA", "LENA", "LEONA", "LESLEY",
    "LILY", "LINDA", "LISA", "LOLA", "LORRAINE", "LOUISE", "LUCIA",
    "LUCY", "LYDIA", "LYNDA", "LYNN",
    # M
    "MADELINE", "MAGGIE", "MARGARET", "MARIA", "MARIANNA", "MARIE",
    "MARINA", "MARTHA", "MARY", "MAYA", "MEGAN", "MELISSA", "MIA",
    "MICHELLE", "MIRANDA", "MOLLY", "MONICA",
    # N
    "NADIA", "NANCY", "NAOMI", "NATALIA", "NATALIE", "NATASHA", "NICOLE",
    "NINA", "NORA",
    # O
    "OLIVIA", "OPHELIA",
    # P
    "PAMELA", "PATRICIA", "PAULA", "PENELOPE", "PETRA", "PHILIPPA",
    "PHOEBE", "PRIYA",
    # R
    "RACHEL", "REBECCA", "REGINA", "RENEE", "RITA", "ROSA", "ROSE",
    "ROSIE", "ROXANNE", "RUBY", "RUTH",
    # S
    "SABRINA", "SAMANTHA", "SANDRA", "SARA", "SARAH", "SCARLETT",
    "SELENA", "SERENA", "SHANNON", "SHARON", "SHEILA", "SIMONE",
    "SOFIA", "SONJA", "SOPHIA", "SOPHIE", "STELLA", "STEPHANIE",
    "SUSAN", "SVETLANA", "SYLVIA",
    # T
    "TAMARA", "TAMMY", "TANYA", "TERESA", "TESSA", "THERESA", "TINA",
    "TORI", "TRACY",
    # U
    "UMA", "URSULA",
    # V
    "VALENTINA", "VALERIA", "VALERIE", "VANESSA", "VERA", "VERONICA",
    "VICTORIA", "VIOLET", "VIRGINIA", "VIVIAN",
    # W
    "WENDY", "WHITNEY",
    # Y
    "YASMIN", "YVETTE", "YVONNE",
    # Z
    "ZOE",
}

_KNOWN_MALE = {
    "AARON", "ADAM", "ALAN", "ALBERT", "ALEX", "ALEXEI", "ALFRED",
    "ANDREW", "ANDY", "ANTON", "ANTONIO", "ARTHUR",
    "BENJAMIN", "BORIS", "BRAD", "BRANDON", "BRIAN", "BRUCE",
    "CARL", "CARLOS", "CHARLES", "CHRIS", "CHRISTIAN", "CHRISTOPHER",
    "COLONEL", "CRAIG",
    "DANIEL", "DAVID", "DENNIS", "DEREK", "DMITRI", "DOMINIC", "DONALD",
    "DOUGLAS",
    "EDGAR", "EDWARD", "ERIC", "ETHAN", "EUGENE",
    "FELIX", "FRANK", "FRED", "FREDERICK",
    "GABRIEL", "GARY", "GEORGE", "GORDON", "GREG", "GREGORY",
    "HARRY", "HENRY", "HUGO",
    "IAN", "IGOR", "IVAN",
    "JACK", "JACOB", "JAMES", "JASON", "JEFF", "JEFFREY", "JEREMY",
    "JOHN", "JONATHAN", "JOSEPH", "JOSH", "JOSHUA", "JULIAN", "JUSTIN",
    "KARL", "KEITH", "KEVIN", "KURT",
    "LANCE", "LARRY", "LEO", "LEON", "LEONARD", "LEWIS", "LIAM", "LOUIS", "LUKE",
    "MARCUS", "MARK", "MARTIN", "MATT", "MATTHEW", "MAX", "MICHAEL",
    "MIGUEL", "MIKE", "MITCHELL",
    "NATHAN", "NEIL", "NICHOLAS", "NICK", "NIKOLAI",
    "OLIVER", "OSCAR",
    "PATRICK", "PAUL", "PETER", "PHILIP",
    "RAYMOND", "RICHARD", "ROBERT", "ROGER", "RONALD", "ROSS", "RYAN",
    "SAMUEL", "SCOTT", "SEAN", "SERGEI", "SIMON", "STEFAN", "STEPHEN",
    "STEVE", "STEVEN",
    "THOMAS", "TIM", "TIMOTHY", "TOM", "TONY",
    "VICTOR", "VIKTOR", "VINCENT", "VLADIMIR", "VLAD",
    "WALTER", "WILLIAM",
    "XAVIER",
    "ZACHARY",
}

# ---------------------------------------------------------------------------
# Name-ending rules  (checked if name not in either dictionary)
# ---------------------------------------------------------------------------

_FEMALE_ENDINGS = (
    "ELLE", "ETTE", "ENNE", "ENNE", "IENNE",
    "INE", "AINE", "EINE",
    "LEEN", "LEEN",
    "ETH",
    "IE",
    "A",          # broad catch-all — e.g. MARINA, VERA, TANYA
)

_MALE_ENDINGS = (
    "SON", "TON", "MAN",
    "ER", "OR", "AR",
    "EY", "AY",
    "ON", "AN", "EN", "IN",
    "EL", "AL", "OL",
    "EK", "IK",
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_female(speaker: str) -> bool:
    """
    Return True if the speaker is female, False if male.
    Works on full speaker strings like "COLONEL KUZNETSOV" or "RACHEL".
    """
    name = _extract_first_name(speaker)

    # 1. Known female dictionary
    if name in _KNOWN_FEMALE:
        return True

    # 2. Known male dictionary
    if name in _KNOWN_MALE:
        return False

    # 3. Female endings
    for ending in _FEMALE_ENDINGS:
        if name.endswith(ending):
            return True

    # 4. Male endings
    for ending in _MALE_ENDINGS:
        if name.endswith(ending):
            return False

    # 5. Default → male
    return False


def is_male(speaker: str) -> bool:
    return not is_female(speaker)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_first_name(speaker: str) -> str:
    """
    Uppercase, strip punctuation, return the first word.
    'Colonel Kuznetsov' → 'COLONEL'  (rank treated as name token)
    'RACHEL'            → 'RACHEL'
    """
    clean = speaker.strip().upper()
    # Remove common rank prefixes so the actual name is checked
    rank_prefixes = {
        "COLONEL", "MAJOR", "GENERAL", "CAPTAIN", "SERGEANT",
        "LIEUTENANT", "AGENT", "DR", "DR.", "MR", "MR.", "MRS", "MRS.",
        "MS", "MS.", "PROF", "PROF.",
    }
    words = clean.split()
    # Skip rank words to get to the actual name
    for word in words:
        w = word.strip(".,")
        if w not in rank_prefixes:
            return w
    return words[0] if words else clean