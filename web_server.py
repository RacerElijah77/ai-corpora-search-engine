# Some boilerplate codes were borrowed from:
#   Conor Bailey's video (https://www.youtube.com/watch?v=dgvLegLW6ek)
import cgi
from http.server import HTTPServer
from http.server import BaseHTTPRequestHandler

import db_build_classifier_dtree
import db_search
import db_view
import math

server_domain = 'localhost'
server_port = 8000

# this will be changed for part 4
top_n = 100
res = []
query = 'Information Retrieval is FUN!'
cla_choice = '0'
top_x_user = '10'
classified_query_label = ''
num_search_results = 0

# template variable -> HTML code from search.html
def generate_search_result(template, position_id):
    # Break up the template into header and footer
    output_beg = template.find(f'<div id="{position_id:s}">') \
                 + len(f'<div id="{position_id:s}">')
    output_header = template[0:output_beg]
    output_end = template.find('</div>', output_beg)
    output_footer = template[output_end:]

    global page_count
    page_count = 0
    # Put the search result in the table form
    output = ''
    output += f'<h4>Classification Label for above query: <span style="color:red;" >{classified_query_label:s}</span></h4>'
    output += '<table id = "td_demo">'
    output += '<thead>'
    output += '  <tr>'
    output += '    <th>Classification Label for this doc</th>'
    output += '    <th>Score</th>'
    output += '    <th>ID</th>'
    output += '    <th>Title</th>'
    output += '    <th>Author</th>'
    output += '    <th>Description</th>'
    output += '    <th>Abstract</th>'
    output += '  </tr>'
    output += '</thead>'
    output += '<tbody>'
    c = ['#dddddd', '#ffffff']
    odd = 0

    # res is only printing x number of times
    # modify HTML links for code

    for r in res:

        if(classified_query_label == r[0]):
            output += f'  <tr style="background-color:{c[odd]:s}">'
            output += f'    <td style="color: green;">{r[0]:s}</td>'
            output += f'    <td>{float(r[1]):.5f}</td>'
            output += f'    <td>{int(r[2]):d}</td>'
            output += f'    <td>{r[3]:s}</td>'
            output += f'    <td>{r[4]:s}</td>'
            output += f'    <td>{r[5]:s}</td>'
            output += f'    <td>{r[6]:s}</td>'
            output += '  </tr>'
            odd += 1
            odd = odd % 2
            page_count += 1

    output += '</tbody>'
    output += '</table>'

    output += f'<h4> Num. of retrieved results: <span style="color:orange;" >{page_count:d}</span></h4>'
    # output += f'<h4>Number of TOTAL docs retrieved (not top X):  <span style="color:green;" >{num_search_results:d}</span></h4>'

    # Debug
    # print(output_header + '\n-----------------------------\n')
    # print(output)
    # print(output_footer + '\n-----------------------------\n')
    return output_header + output + output_footer


class RoscoeRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('/search'):
            self.send_response(200)
            self.send_header('content-type', 'text/html')
            self.end_headers()

            # this is reading the HTML file
            with open('html/search.html', 'rt') as f:
                output = f.read()

            pos = output.find('</textarea>')

            output = output[0:pos] + query + output[pos:]
            output = generate_search_result(output, 'result-list')

        else:
            self.send_response(200)
            self.send_header('content-type', 'text/html')
            self.end_headers()

            with open('html/index.html', 'rt') as f:
                output = f.read()

        self.wfile.write(output.encode())


    def do_POST(self):
        global res, query
        if self.path.endswith('/search'):
            ctype, pdict = cgi.parse_header(self.headers.get('content-type'))
            pdict['boundary'] = bytes(pdict['boundary'], 'utf-8')
            # content_len = int(self.headers.get('Content-length'))
            # pdict['CONTENT-LENGTH'] = content_len
            if ctype == 'multipart/form-data':
                fields = cgi.parse_multipart(self.rfile, pdict)
                query = fields.get('query')

                global cla_choice
                cla_choice = fields.get('cla_choice')[0]
                print("CLA CHOICE: " + str(cla_choice))

                deletion_choice = fields.get('deletion_option')[0]
                print("DELETION VALUE: " + str(deletion_choice))

                global  top_n
                if(top_x_user == ''):
                    top_n = 10
                else:
                    top_n = int(top_x_user)

                # query item above is coming in as a list, so we need to take [0]
                query = query[0]

                # NEW FUNCTION testing new classifier function, add this to HTML and call python script
                db_build_classifier_dtree.select_clf_algorithm(cla_choice, deletion_choice)

                # NEW FUNCTION -> Classify query based on 20 labels and display variable in HTML result
                global classified_query_label
                classified_query_label = db_build_classifier_dtree.classify_query(query)

                # Search the database and update the session result
                global acc
                acc = db_search.search(query)
                # print(acc)

                global num_search_results
                num_search_results = len(acc)

                # display only top 10 results by default
                # db_view.pagination_feature(acc, top_n)

                # changed top_n to num_search_results
                res = db_view.get_result_ait_top_n(acc, num_search_results)

            self.send_response(301)
            self.send_header('content-type', 'text/html')
            self.send_header('Location', '/search')
            self.end_headers()


def main():
    server_address = (server_domain, server_port)
    server = HTTPServer(server_address, RoscoeRequestHandler)
    print('HTTP server ' + server_domain + f' running on port {server_port: d}')
    server.serve_forever()


main()
