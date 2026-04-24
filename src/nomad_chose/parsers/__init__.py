from nomad.config.models.plugins import ParserEntryPoint
from pydantic import Field

class ChoseJVParserEntryPoint(ParserEntryPoint):
    """
    Entry point for the CHOSE JV CSV parser.

    Matches files whose name ends with .jv.csv or contains _JV_,
    and whose first non-comment line is the header 'voltage,current_density'.
    The content check prevents false positives from other CSV files.
    """
    parameter: int = Field(0, description='Custom configuration parameter')
    def load(self):
        from nomad_chose.parsers.jv_parser import ChoseJVParser
        return ChoseJVParser(**self.model_dump())


parser_entry_point = ChoseJVParserEntryPoint(
    name='ChoseJVParser',
    description='Parser for JV measurement CSV files from the CHOSE lab instrument.',
    # File name matcher — adjust the pattern to your actual naming convention
    mainfile_name_re=r'.*\.(jv\.csv|JV\.csv)|.*_[Jj][Vv]_.*\.csv',
    # Content check: file must contain the exact column header
    mainfile_contents_re=r'voltage,current_density',
)