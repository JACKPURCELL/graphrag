import os
import subprocess
import networkx as nx
import json
import webbrowser
import http.server
import socketserver
import threading

# 读取GraphML文件并转换为JSON
def graphml_to_json(graphml_file):
    G = nx.read_graphml(graphml_file)
    data = nx.node_link_data(G)
    return json.dumps(data)


# 创建HTML文件
def create_html(html_path):
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }
        svg {
            width: 100%;
            height: 100%;
        }
        .links line {
            stroke: #999;
            stroke-opacity: 0.6;
        }
        .nodes circle {
            stroke: #fff;
            stroke-width: 1.5px;
        }
        .node-label {
            font-size: 12px;
            pointer-events: none;
        }
        .link-label {
            font-size: 10px;
            fill: #666;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .link:hover .link-label {
            opacity: 1;
        }
        .tooltip {
            position: absolute;
            text-align: left;
            padding: 10px;
            font: 12px sans-serif;
            background: lightsteelblue;
            border: 0px;
            border-radius: 8px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
            max-width: 300px;
        }
        .legend {
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 5px;
        }
        .legend-item {
            margin: 5px 0;
        }
        .legend-color {
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 5px;
            vertical-align: middle;
        }
        #search-container {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div id="search-container">
        <input type="text" id="search-input" placeholder="Search node...">
        <button id="search-button">Search</button>
        <button id="prev-button">Previous</button>
        <button id="next-button">Next</button>
        <span id="search-status"></span>
    </div>
    <svg></svg>
    <div class="tooltip"></div>
    <div class="legend"></div>
    <script type="text/javascript" src="./graph_json.js"></script>
    <script>
        const graphData = graphJson;
        
        const svg = d3.select("svg"),
            width = window.innerWidth,
            height = window.innerHeight;

        svg.attr("viewBox", [0, 0, width, height]);

        const g = svg.append("g");

        const entityTypes = [...new Set(graphData.nodes.map(d => d.entity_type))];
        const color = d3.scaleOrdinal(d3.schemeCategory10).domain(entityTypes);

        const simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(150))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collide", d3.forceCollide().radius(30));

        const linkGroup = g.append("g")
            .attr("class", "links")
            .selectAll("g")
            .data(graphData.links)
            .enter().append("g")
            .attr("class", "link");

        const link = linkGroup.append("line")
            .attr("stroke-width", d => Math.sqrt(d.value));

        const linkLabel = linkGroup.append("text")
            .attr("class", "link-label")
            .text(d => d.description || "");

        const node = g.append("g")
            .attr("class", "nodes")
            .selectAll("circle")
            .data(graphData.nodes)
            .enter().append("circle")
            .attr("r", 5)
            .attr("fill", d => color(d.entity_type))
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        const nodeLabel = g.append("g")
            .attr("class", "node-labels")
            .selectAll("text")
            .data(graphData.nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .text(d => d.id);

        const tooltip = d3.select(".tooltip");

        node.on("mouseover", function(event, d) {
            tooltip.transition()
                .duration(200)
                .style("opacity", .9);
            tooltip.html(`<strong>${d.id}</strong><br>Entity Type: ${d.entity_type}<br>Description: ${d.description || "N/A"}`)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", function(d) {
            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
        });

        const legend = d3.select(".legend");
        entityTypes.forEach(type => {
            legend.append("div")
                .attr("class", "legend-item")
                .html(`<span class="legend-color" style="background-color: ${color(type)}"></span>${type}`);
        });

        simulation
            .nodes(graphData.nodes)
            .on("tick", ticked);

        simulation.force("link")
            .links(graphData.links);

        function ticked() {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            linkLabel
                .attr("x", d => (d.source.x + d.target.x) / 2)
                .attr("y", d => (d.source.y + d.target.y) / 2)
                .attr("text-anchor", "middle")
                .attr("dominant-baseline", "middle");

            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);

            nodeLabel
                .attr("x", d => d.x + 8)
                .attr("y", d => d.y + 3);
        }

        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }

        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }

        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }

        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on("zoom", zoomed);

        svg.call(zoom);

        function zoomed(event) {
            g.attr("transform", event.transform);
        }

        // Search functionality
        let currentMatchIndex = -1;
        let matches = [];

        function highlightNodeAndConnections(index) {
            if (matches.length === 0) return;

            const foundNode = matches[index];

            // Highlight found node
            node.attr('r', d => d.id === foundNode.id ? 10 : 5)
                .attr('stroke', d => d.id === foundNode.id ? 'red' : '#fff')
                .attr('stroke-width', d => d.id === foundNode.id ? 3 : 1.5);

            // Highlight connected links and nodes
            link.attr('stroke', d => (d.source.id === foundNode.id || d.target.id === foundNode.id) ? 'red' : '#999')
                .attr('stroke-opacity', d => (d.source.id === foundNode.id || d.target.id === foundNode.id) ? 1 : 0.6);

            node.attr('fill', d => {
                if (d.id === foundNode.id) return 'red';
                const connected = graphData.links.some(l => (l.source.id === foundNode.id && l.target.id === d.id) || (l.target.id === foundNode.id && l.source.id === d.id));
                return connected ? 'orange' : color(d.entity_type);
            });

            // Zoom and center onto the found node
            const scale = 1.5;
            const translate = [width / 2 - scale * foundNode.x, height / 2 - scale * foundNode.y];
            svg.transition().duration(750).call(
                zoom.transform,
                d3.zoomIdentity.translate(...translate).scale(scale)
            );

            document.getElementById('search-status').textContent = `Match ${index + 1} of ${matches.length}`;
        }

        document.getElementById('search-button').addEventListener('click', () => {
            const searchTerm = document.getElementById('search-input').value.trim().toLowerCase();
            matches = graphData.nodes.filter(n => n.id.toLowerCase().includes(searchTerm));

            if (matches.length > 0) {
                currentMatchIndex = 0;
                highlightNodeAndConnections(currentMatchIndex);
            } else {
                alert('No matches found!');
                document.getElementById('search-status').textContent = '';
            }
        });

        document.getElementById('next-button').addEventListener('click', () => {
            if (matches.length > 0) {
                currentMatchIndex = (currentMatchIndex + 1) % matches.length;
                highlightNodeAndConnections(currentMatchIndex);
            }
        });

        document.getElementById('prev-button').addEventListener('click', () => {
            if (matches.length > 0) {
                currentMatchIndex = (currentMatchIndex - 1 + matches.length) % matches.length;
                highlightNodeAndConnections(currentMatchIndex);
            }
        });

    </script>
</body>
</html>


"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def create_json(json_data, json_path):
    json_data = "var graphJson = " + json_data.replace('\\"', '').replace("'", "\\'").replace("\n", "")
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(json_data)

def kill_process_on_port(port):
    try:
        # 查找占用端口的进程号 (PID)
        result = subprocess.run(['lsof', '-t', f'-i:{port}'], capture_output=True, text=True)
        pid = result.stdout.strip()

        if pid:
            print(f"Port {port} is in use by process {pid}. Killing process...")
            subprocess.run(['kill', '-9', pid])
            print(f"Process {pid} killed.")
        else:
            print(f"Port {port} is free.")
    except Exception as e:
        print(f"Error checking/killing process on port {port}: {e}")

# 启动简单的HTTP服务器并检查端口
def start_server():
    PORT = 8000
    
    # Kill any process using the PORT
    kill_process_on_port(PORT)
    
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Server started at http://localhost:{PORT}")
        httpd.serve_forever()

# # 启动简单的HTTP服务器
# def start_server():
#     handler = http.server.SimpleHTTPRequestHandler
#     with socketserver.TCPServer(("", 8000), handler) as httpd:
#         print("Server started at http://localhost:8000")
#         httpd.serve_forever()

# # Main function to start server
# def start_server():
#     PORT = 8000
    
#     # Kill any process using the PORT
#     kill_process_on_port(PORT)
    
#     Handler = http.server.SimpleHTTPRequestHandler
    
#     with socketserver.TCPServer(("", PORT), Handler) as httpd:
#         print(f"Serving at port {PORT}")
#         httpd.serve_forever()

# 主函数
def visualize_graphml(graphml_file, html_path,json_path):
    json_data = graphml_to_json(graphml_file)
    create_json(json_data, json_path)
    create_html(html_path)
    # 在后台启动服务器
    # server_thread = threading.Thread(target=start_server)
    # server_thread.daemon = True
    # server_thread.start()
    #socketserver.TCPServer.allow_reuse_address = True
    # 打开默认浏览器
    # webbrowser.open('http://localhost:8000/' + html_path)
    
    # print("Visualization is ready. Press Ctrl+C to exit.")
    # try:
    #     # 保持主线程运行
    #     while True:
    #         pass
    # except KeyboardInterrupt:
    #     print("Shutting down...")

# 使用示例
if __name__ == "__main__":
    # graphml_file = "ragtest7_cyber_small/output/20240919-112853/artifacts/merged_graph.graphml"  # 替换为您的GraphML文件路径
    # graphml_file = "ragtest/output/20240805-181540/artifacts/merged_graph.graphml"
    # graphml_file = "ragtest6_modify/output/20240905-093528/artifacts/merged_graph.graphml"
    base_path = '/home/ljc/data/graphrag'
    graphml_file = "/data/yuhui/6/graphrag/ragtest8_medical_small/output/20240927-164809/artifacts/merged_graph.graphml"
    html_path = base_path + "/graph_visualization2.html"
    json_path = base_path + "/graph_json.js"
    visualize_graphml(graphml_file, html_path,json_path)