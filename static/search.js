
    // 搜索功能
    document.addEventListener('DOMContentLoaded', function() {
        const searchInput = document.getElementById('search-input');
        const searchResults = document.getElementById('search-results');
        
        if (!searchInput || !searchResults) return;
        
        // 加载函数数据
        fetch('search_data.json')
            .then(response => response.json())
            .then(functions => {
                // 搜索函数
                searchInput.addEventListener('input', function() {
                    const query = this.value.toLowerCase().trim();
                    
                    if (query.length < 2) {
                        searchResults.style.display = 'none';
                        return;
                    }
                    
                    // 过滤函数
                    const filtered = functions.filter(func => {
                        return func.name.toLowerCase().includes(query) || 
                               func.description.toLowerCase().includes(query);
                    });
                    
                    // 显示结果
                    if (filtered.length > 0) {
                        searchResults.innerHTML = '';
                        filtered.forEach(func => {
                            const resultItem = document.createElement('div');
                            resultItem.className = 'search-result-item';
                            resultItem.innerHTML = `<strong>${func.name}</strong> - ${func.description.substring(0, 100).replace(/<\/?[^>]+(>|$)/g, "")}${func.description.length > 100 ? '...' : ''}`;
                            resultItem.onclick = function() {
                                window.location.href = `${func.name}.html`;
                            };
                            searchResults.appendChild(resultItem);
                        });
                        searchResults.style.display = 'block';
                    } else {
                        searchResults.innerHTML = '<div class="search-result-item">没有找到相关函数</div>';
                        searchResults.style.display = 'block';
                    }
                });
                
                // 点击外部关闭搜索结果
                document.addEventListener('click', function(e) {
                    if (e.target !== searchInput && !searchResults.contains(e.target)) {
                        searchResults.style.display = 'none';
                    }
                });
                
                // ESC键关闭搜索结果
                document.addEventListener('keydown', function(e) {
                    if (e.key === 'Escape') {
                        searchResults.style.display = 'none';
                    }
                });
            })
            .catch(error => console.error('加载搜索数据失败:', error));
    });
    