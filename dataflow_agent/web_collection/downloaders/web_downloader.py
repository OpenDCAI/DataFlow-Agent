"""
Web Downloader
~~~~~~~~~~~~~~

通用网页内容下载器
"""

import os
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse, urljoin
from dataflow_agent.logger import get_logger

logger = get_logger(__name__)


class WebDownloader:
    """Generic web content downloader"""
    
    def __init__(
        self, 
        download_dir: str = "./downloads",
        timeout: int = 60,
    ):
        """
        Initialize Web downloader
        
        Args:
            download_dir: Directory to save downloads
            timeout: Request timeout in seconds
        """
        self.download_dir = download_dir
        self.timeout = timeout
        os.makedirs(download_dir, exist_ok=True)
        
        # Check if aiohttp is available
        try:
            import aiohttp
            self.aiohttp = aiohttp
            self.available = True
        except ImportError:
            logger.warning("aiohttp not installed")
            self.aiohttp = None
            self.available = False
    
    async def download_file(
        self,
        url: str,
        filename: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Download a file from URL
        
        Args:
            url: URL to download from
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Tuple of (download_path, error_message)
        """
        if not self.available:
            return None, "aiohttp not installed"
        
        try:
            # Generate filename from URL if not provided
            if not filename:
                parsed = urlparse(url)
                filename = os.path.basename(parsed.path) or "download"
                # Add extension if missing
                if '.' not in filename:
                    filename += '.html'
            
            filepath = os.path.join(self.download_dir, filename)
            
            async with self.aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.timeout) as response:
                    if response.status == 200:
                        content = await response.read()
                        with open(filepath, 'wb') as f:
                            f.write(content)
                        logger.info(f"Downloaded {url} to {filepath}")
                        return filepath, None
                    else:
                        return None, f"HTTP {response.status}"
            
        except asyncio.TimeoutError:
            return None, "Request timeout"
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None, str(e)
    
    async def fetch_page_content(
        self,
        url: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Fetch page content as text
        
        Args:
            url: URL to fetch
            
        Returns:
            Tuple of (content, error_message)
        """
        if not self.available:
            return None, "aiohttp not installed"
        
        try:
            async with self.aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.timeout) as response:
                    if response.status == 200:
                        content = await response.text()
                        return content, None
                    else:
                        return None, f"HTTP {response.status}"
            
        except asyncio.TimeoutError:
            return None, "Request timeout"
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None, str(e)
    
    async def extract_text_from_html(
        self,
        html_content: str,
    ) -> str:
        """
        Extract text content from HTML
        
        Args:
            html_content: HTML string
            
        Returns:
            Extracted text content
        """
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = '\n'.join(lines)
            
            return text
            
        except ImportError:
            logger.warning("beautifulsoup4 not installed, returning raw HTML")
            return html_content
        except Exception as e:
            logger.error(f"Failed to extract text from HTML: {e}")
            return html_content
    
    async def crawl_links(
        self,
        url: str,
        max_depth: int = 1,
        max_links: int = 10,
    ) -> List[str]:
        """
        Crawl and extract links from a page
        
        Args:
            url: Starting URL
            max_depth: Maximum crawl depth
            max_links: Maximum number of links to return
            
        Returns:
            List of discovered URLs
        """
        if not self.available:
            return []
        
        discovered_urls = []
        
        try:
            content, error = await self.fetch_page_content(url)
            if error or not content:
                return []
            
            try:
                from bs4 import BeautifulSoup
                
                soup = BeautifulSoup(content, 'html.parser')
                
                # Find all links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    
                    # Convert relative URLs to absolute
                    full_url = urljoin(url, href)
                    
                    # Filter out non-HTTP URLs
                    if full_url.startswith(('http://', 'https://')):
                        # Skip common non-content URLs
                        skip_patterns = [
                            '/login', '/signup', '/register',
                            '/terms', '/privacy', '/contact',
                            '.js', '.css', '.png', '.jpg', '.gif',
                        ]
                        if not any(p in full_url.lower() for p in skip_patterns):
                            discovered_urls.append(full_url)
                
                # Remove duplicates and limit
                discovered_urls = list(set(discovered_urls))[:max_links]
                
            except ImportError:
                logger.warning("beautifulsoup4 not installed")
            
        except Exception as e:
            logger.error(f"Failed to crawl links from {url}: {e}")
        
        return discovered_urls
