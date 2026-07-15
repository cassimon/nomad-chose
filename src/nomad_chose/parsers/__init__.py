from nomad.config.models.plugins import ParserEntryPoint
from pydantic import Field


class NewParserEntryPoint(ParserEntryPoint):
    parameter: int = Field(0, description='Custom configuration parameter')

    def load(self):
        from nomad_chose.parsers.parsers import ChoseParser

        return ChoseParser(**self.model_dump())


parser_entry_point = NewParserEntryPoint(
    name='ChoseParser',
    description='Unified parser for JV, Stability, and IPCE files from the CHOSE setup.',
    mainfile_name_re=r'.*\.(csv|txt)$',
    mainfile_contents_re=(
        r'voltage,current_density'
        r'|Test\tStability \(JV\)'
        r'|Test\tStability \(Parameters\)'
        r'|Test\tStability \(Tracking\)'
        r'|Test\tIPCE'
        r'|Wavelength nm\. T%'
    ),
)

class ChoseJVParserEntryPoint(ParserEntryPoint):
    """
    Entry point for the CHOSE JV CSV parser.

    Matches files whose name ends with .jv.csv or contains _JV_,
    and whose first non-comment line is the header 'voltage,current_density'.
    The content check prevents false positives from other CSV files.
    """

    def load(self):
        from nomad_chose.parsers.parsers import ChoseParser
        return ChoseParser(**self.model_dump())


chose_jv_parser = ChoseJVParserEntryPoint(
    name='ChoseJVParser',
    description='Backward compatible alias of the unified CHOSE parser.',
    mainfile_name_re=parser_entry_point.mainfile_name_re,
    mainfile_contents_re=parser_entry_point.mainfile_contents_re,
)